import matplotlib
#matplotlib.use('Agg')
from photutils import detect_sources
from photutils import detect_threshold
from photutils import deblend_sources
from astropy.convolution import Tophat2DKernel,Gaussian2DKernel, Box2DKernel,convolve, convolve_fft
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils import source_properties
from photutils.segmentation import make_source_mask
from scipy import ndimage
from matplotlib.colors import LogNorm
#import Mask4asymm_YangLi as Asymmask
from matplotlib import patches
import numpy as np
import numpy.ma as ma
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import subprocess
from astropy.stats import gaussian_fwhm_to_sigma
import csv

warnings.filterwarnings("ignore")
# Reference: Sazonova et al. 2021


# -----------------------------------------------------------------------------------------------------------------
# Parameters:
# -----------------------------------------------------------------------------------------------------------------

def params(fwhm0,path0,fitsname0,z0,Rpet,pixelSize0,cutlength=None,targposi0=None,target0=None):
    global fwhm,path,fitsname,target,Rpet_asec,Rp,pixelSize,targposi,exptime,z_targ
    Rpet_asec = Rpet
    fwhm = float(fwhm0)
    pixelSize = float(pixelSize0)
    Rp = Rpet_asec/pixelSize
    path=str(path0)
    fitsname=str(fitsname0)
    z_targ = z0
    
#    ra = float(ra0)
#    dec = float(dec0)

    data_org = fits.getdata(path+fitsname)
    header = fits.getheader(path+fitsname)
    exptime = header['EXPTIME']
    ny, nx = data_org.shape

    if targposi0 is None:
        targposi=(nx/2.,ny/2.)
    else:
        targposi=targposi0

    if cutlength is None:
        data = data_org
    else:
        L = int(cutlength/2.)
        xmin = int(max(targposi[0]-L, 0))
        xmax = int(min(targposi[0]+L,nx))
        ymin = int(max(targposi[1]-L, 0))
        ymax = int(min(targposi[1]+L,ny))
        data = data_org[ymin:ymax,xmin:xmax]

    if target0 is None:
        target=fitsname[:-5]
    else:
        target=str(target0)
    return data

def elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh=None):
    x=xmesh-posi[0]
    b=a*(1.-ellipticity)
    theta=PA*np.pi/180.
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    if ymesh is not None:
        y=ymesh-posi[1]
        return A*x**2+B*x*y+C*y**2-(a**2)*(b**2)
    else:
        coeff = [C, B*x, A*x**2-(a**2)*(b**2)]
        return np.roots(coeff)+posi[1]

def Maskellipse (mask,posi,a,ellipticity,PA,antimask=False):
    #PA from x axis anti-clockwise, degree
    masktemp=mask.copy()
    ny,nx=mask.shape
    xaxis = np.arange(nx)
    yaxis = np.arange(ny)
    xmesh, ymesh = np.meshgrid(xaxis, yaxis)
    if not antimask:
        masktemp[elliptical_func(xmesh+0.5,posi,a,ellipticity,PA,ymesh+0.5)<0]=1.
    else:
        masktemp[elliptical_func(xmesh+0.5,posi,a,ellipticity,PA,ymesh+0.5)<0]=0.
    return masktemp.astype(mask.dtype)




# -----------------------------------------------------------------------------------------------------------------
# Run SExtractor:
# -----------------------------------------------------------------------------------------------------------------

def run_SE(det_thresh, det_area=5, basic_config='General.txt', checkimage_type='SEGMENTATION'):
    label = str(checkimage_type[:4])
    configtype = str(basic_config[:4])
    
    cmd=str('sex {0}{1} -c {7} -DETECT_THRESH {5} -DETECT_MINAREA {6} -CHECKIMAGE_TYPE {8} -CHECKIMAGE_NAME {0}{2}_{3}{4}.fits -CATALOG_NAME {0}{2}_{3}.cat'.format(path,fitsname,target,configtype,label,det_thresh,det_area,basic_config,str(checkimage_type)))
    #print(cmd)
    duang = subprocess.check_call(cmd, shell=True)
    while checkimage_type=='SEGMENTATION' and duang==0:
        segmimg = fits.getdata('{0}{2}_{3}{4}.fits'.format(path,fitsname,target,configtype,label))
        targID = segmimg[int(targposi[1]),int(targposi[0])]
        if targID>0:
            print(duang)
            break
        elif det_thresh<=1:
            print('No centroid target found even threshold<=1! You can modify the DETECT_MINAREA or Kernel.')
            break
        else:
            print('Threshold={0} failed to find the target! Try lower threshold:'.format(det_thresh) )
            det_thresh=max(det_thresh-0.5, 1)
            cmdnew=str('sex {0}{1} -c {7} -DETECT_THRESH {5} -DETECT_MINAREA {6} -CHECKIMAGE_NAME {0}{2}_{3}{4}.fits -CATALOG_NAME {0}{2}_{3}.cat'.format(path,fitsname,target,configtype,label,det_thresh,det_area,basic_config))
            duang = subprocess.check_call(cmdnew, shell=True)
            print(str(det_thresh))
    SEcheckname='{2}_{3}{4}.fits'.format(path,fitsname,target,configtype,label)
    SEcatname='{2}_{3}.cat'.format(path,fitsname,target,configtype,label)
    #print('SE files: '+str(SEcatname)+','+str(SEcheckname))

    return SEcatname,SEcheckname


def Mask_correction(image, mexhat_cat, segmMex, maskgrowth=True,Nconv=1, gal = False):
    #This correction is aiming to mask bright foreground stars inside 2 Rp of target galaxy.
    srcXp = targposi[0]
    srcYp = targposi[1]

#    global mag_comp_min
    
    List_pair_comp = [999]
    
    flag_merger0 = 0
    mag_comp = -999
    ncomp_gal0 = 0
    ncomp_pair0 = 0
    
    skymask = make_source_mask(image, nsigma=3, npixels=5, dilate_size=11)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)  #, mask=skymask
    
    stars = ascii.read(path + mexhat_cat)
    props=stars[:]
    #plt.figure(figsize=(6,6))
    
    segm_targ  = segmMex[int(srcXp),int(srcYp)]
    print('Target segm ID:',segm_targ)
    
    maskseg0=np.zeros_like(image, dtype=bool)
    maskstar=np.zeros_like(image, dtype=bool)
    maskstar[np.isnan(image)] = True
    maskstar[np.isinf(image)] = True
    
    #plt.imshow(image,norm=LogNorm(0.005),cmap='Greys', interpolation='nearest', origin='lower')
    #ax=plt.gca()
    
    found=True
    Tnumber=None
    header0 = 0
 
 # Find the target:
    for prop in props:
        position = (prop['X_IMAGE'], prop['Y_IMAGE'])
        a = prop['A_IMAGE']
        b = prop['B_IMAGE']
        theta = prop['THETA_IMAGE']
        petro = prop['PETRO_RADIUS']
        ID = prop['NUMBER']
        targID = []

        if ID == segm_targ: #maskseg0[int(srcYp),int(srcXp)]&found:
            global magCISO,fluxCISO
            aC=a*petro
            bC=b*petro
            thetaC=theta
            Tnumber = ID
            print(Tnumber,'target num1')
            ra0 = prop['X_WORLD']
            dec0 = prop['Y_WORLD']
            x_targ = prop['X_IMAGE']
            y_targ = prop['Y_IMAGE']
            fluxCP = prop['FLUX_MAX']
            fluxCISO = prop['FLUX_ISOCOR']
            magCISO = -2.5*np.log10(fluxCISO/exptime)+24.74  # i band magzp
            found=False
            break
        else:
            continue
    if found is True:
        fluxCISO = -999
        print('No centroid galaxy detected by Mexhat!')
    
    for prop in props:
        position = (prop['X_IMAGE'], prop['Y_IMAGE'])
        a = prop['A_IMAGE']
        b = prop['B_IMAGE']
        ra = prop['X_WORLD']
        dec = prop['Y_WORLD']
        x = prop['X_IMAGE']
        y = prop['Y_IMAGE']
        theta = prop['THETA_IMAGE']
        fluxPeak = prop['FLUX_MAX']
        petro = prop['PETRO_RADIUS']
        kind = prop['CLASS_STAR']
        fluxISO = prop['FLUX_ISOCOR']
        magISO = -2.5*np.log10(fluxISO/exptime)+24.74  # i band magzp
        #        A = 2.91 * ( ( fluxISO/(26*sky_std*10) )**(1/2.26) - 1 )**0.5  #v1
        A = 1.98 * ( ( fluxISO/(27*sky_std*10) )**(1/1.618) - 1 )**0.5
        Dist = np.sqrt((srcXp-position[0])**2+(srcYp-position[1])**2)
        if prop['NUMBER']==Tnumber:
            print(Tnumber,'targ number')
            continue
        if prop['CLASS_STAR']<0.75 and Dist < 2*Rp and gal == True: # ,  -2.5*np.log10(fluxISO)+24.74 < 20
            # -----------------------------------------------------------------------------------------------------------------
            # Generate comp file: Only when there is companion.
            # -----------------------------------------------------------------------------------------------------------------
            if header0 == 0 :
                file_cp = path+target+'_comp.csv'
                open(file_cp, 'w') .close

                header = ['target', 'RA_targ', 'DEC_targ', 'Rp_asec', 'z_targ','x_targ', 'y_targ', 'mag_targ', 'comp_id', 'flag_majorcomp','x_comp', 'y_comp', 'mag_comp', 'class_comp', 'sep_asec', 'RA_comp', 'DEC_comp','a_comp', 'b_comp' ]

                with open(file_cp, 'a+', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write the header
                    writer.writerow(header)
            # -----------------------------------------------------------------------------------------------------------------
            header0 += 1
            mag_comp = magISO
            class_comp = kind
            dist_comp = Dist
            

            ra_comp = ra
            dec_comp = dec
            x_comp = x
            y_comp = y
            a_comp = a
            b_comp = b
            ncomp_gal0 += 1
            
            if fluxISO>fluxCISO*0.25:
                flag_merger0 = 1
                flag_majcomp = 1
                mag_pair = mag_comp
                List_pair_comp.append(mag_pair)
                if gal == True:
                    ncomp_pair0 +=1
            else:
                flag_majcomp = 0
                magnif = 3
                maskstar=Maskellipse(maskstar,position,magnif*a,(1-b/a),theta)
                #maskstar = maskRs['mask']

            data = [target, ra0, dec0, format(Rpet_asec,'.4f'), z_targ,x_targ, y_targ, format(magCISO,'.4f'), ncomp_gal0,flag_majcomp, x_comp, y_comp, format(mag_comp,'.4f'), class_comp, format(dist_comp,'.4f'), ra_comp, dec_comp, a_comp,b_comp]

            with open(file_cp, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
#
#        elif prop['CLASS_STAR']<0.75 and Dist > 2*Rp and gal == True:
#            magnif = 3
#            print(prop['NUMBER'])
#            maskstar=Maskellipse(maskstar,position,magnif*a,(1-b/a),theta)
#
#            #maskstar = maskRs['mask']

        elif prop['CLASS_STAR']>0.75 and np.isnan(A)==0 and gal == False:
            maskstar=Maskellipse(maskstar,position,A*a,(1-b/a),theta)
        
        else:
            continue


    if gal == True:
        global flag_merger,ncomp_gal,ncomp_pair
        flag_merger = flag_merger0
        ncomp_gal =  ncomp_gal0
        ncomp_pair = ncomp_pair0
        
        global mag_comp_min
        mag_comp_min0 = min(List_pair_comp)
        #    print('mag_min:',mag_comp_min0)
        if mag_comp_min0 == 999:
            mag_comp_min = -999
        else:
            mag_comp_min = mag_comp_min0

    
    
    return maskstar






# -----------------------------------------------------------------------------------------------------------------
# Mask:
# -----------------------------------------------------------------------------------------------------------------

def HotCoolCold(image, posi0, snr = 'SNR1', tophatSize=5, nsigcold=1,npixcold=5): #npixcold=None,hotperc=0.99, nsigcool=1,npixhot=1, npixcool=None,
    print(snr)
    skymask = make_source_mask(image, nsigma=3, npixels=5, dilate_size=11)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, maxiters=5,mask=skymask)
    
    imagenew1 = image.copy()
    imagenew1[np.isnan(image)] = np.ma.masked
    imagenew1[np.isinf(image)] = np.ma.masked
    imagenew = imagenew1 - sky_median
    
#    print('sky_std:',sky_std)

#    Rp = Rpet/pixelSize
    ny,nx=image.shape
    posi = [int(posi0[0]),int(posi0[1])]


    OUT3 = run_SE(det_thresh=1, det_area=5, basic_config='Deb3_General.txt')  # deblend_contrast = 0.001
    outcatname3=OUT3[0]
    outsegname3=OUT3[1]
    segmOUT3 = fits.getdata(path+outsegname3)

    OUT4 = run_SE(det_thresh=1, det_area=5, basic_config='Deb4_General.txt')  # deblend_contrast = 1
    outcatname4=OUT4[0]
    outsegname4=OUT4[1]
    segmOUT4 = fits.getdata(path+outsegname4)

# ***** Cold mode: *****

    segm_cold = segmOUT4
    targ_cold = segm_cold[posi[0],posi[1]]

    segm_deblend_cold = segmOUT3
    targ_deb_cold = segm_deblend_cold[posi[0],posi[1]]


# ***** Mask contaminants outside 2Rp through cold mode segm. *****
    ellipticity=0
    PA=90
    a = 2*Rp
    
    segm_phot = segm_deblend_cold.copy()
    glxs = ascii.read(path + outcatname3)
    props_deblend_cold = glxs[:]
    for prop in props_deblend_cold:
        ID = prop['NUMBER']
        position = (prop['X_IMAGE'],prop['Y_IMAGE'])
        if ID == targ_deb_cold:
            segm_phot[segm_phot==ID] = 0
        elif ((position[0]-posi0[0])**2+(position[1]-posi0[1])**2) / (a**2) < 1.:
            segm_phot[segm_phot==ID] = 0
        else:
            segm_phot[segm_phot==ID] = 1


    maskcold_out = np.zeros_like(image, dtype=bool)
    maskcold_out[np.isnan(image)] = True
    maskcold_out[np.isinf(image)] = True
    maskcold_out[segm_phot==1]=1

# *****  Convolve the 2Rpet outer part with a kernel on cool mask *****

    sigma = 3*fwhm*gaussian_fwhm_to_sigma

    kernel = Gaussian2DKernel(int(sigma))
    kernel.normalize()
#
#    kernel = Tophat2DKernel(100)
#    kernel.normalize()

    convcoldoutmask = convolve_fft(maskcold_out,kernel)
    maskconv_out = convcoldoutmask > np.nanmean(convcoldoutmask.flatten())
    maskout = np.logical_or(maskcold_out, maskconv_out)



# *****  Mask saturate stars *****
    masksaturate = np.zeros_like(image,dtype=bool)
    saturate_flux=[]
    flag_satu0 = 0

    satustar = ascii.read(path + outcatname4)
    props_cold = satustar[:]
    for prop in props_cold:
        position = (prop['X_IMAGE'],prop['Y_IMAGE'])
        area = prop['ISOAREA_IMAGE']
        saturate_flux = prop['FLUX_ISOCOR']
        satu_a = prop['A_IMAGE']
        satu_b = prop['B_IMAGE']
        satu_theta = prop['THETA_IMAGE']
        keep = prop['NUMBER']
        if prop['NUMBER'] == targ_cold:
            continue
        elif area > 400:
            indivmask = np.zeros_like(image)
            indivmask[segm_cold==keep]=1
            convindiv0 = convolve_fft(indivmask,kernel)
            convindiv = convindiv0 > np.nanmean(convindiv0.flatten())
            indivmask = ~convindiv
            indiv = ma.array(image,mask=indivmask)
            contain_nan = (True in np.isnan(indiv))
        
            if contain_nan==True and saturate_flux > 1e4:
                print('Saturate ...............', position )
                magnif = np.sqrt(area)*0.07
                nanposi = np.argwhere(np.isnan(indiv))[0]
                print(nanposi)
                masksaturate=Maskellipse(masksaturate,(nanposi[1],nanposi[0]), magnif*satu_b,0,satu_theta)
                #masksaturate=maskRp0['mask']
                flag_satu0 = 1
            else:
                continue
    print('Ending saturate mask.')
    global flag_satu
    flag_satu = flag_satu0


    # Central mask: only use SExtractor mask
    #************** Mask all stars and outer galaxy ***********************

    Mex = run_SE(det_thresh=1, det_area=5, basic_config='Deb1_General.txt')  # det_thresh=7
    mexcatname=Mex[0]
    mexsegname=Mex[1]
    segmMex = fits.getdata(path+mexsegname)
    mask_se_star = Mask_correction(imagenew,mexcatname, segmMex)

    Mex2 = run_SE(det_thresh=1, det_area=5, basic_config='Deb2_General.txt')  # det_thresh=7
    mexcatname2=Mex2[0]
    mexsegname2=Mex2[1]
    segmMex2 = fits.getdata(path+mexsegname2)
    mask_se_gal = Mask_correction(imagenew,mexcatname2, segmMex2,gal = True)

    
    
    mask_cat = np.logical_or(mask_se_star,mask_se_gal)
    mask_segm = np.logical_or(maskout,masksaturate)
    maskfinal = np.logical_or(mask_cat,mask_segm)

    ny, nx = image.shape
    mask2total = maskfinal.flatten().sum()/(nx*ny)
#    print('m2t:',mask2total)

    hdu = fits.PrimaryHDU(maskfinal.astype('float64'))
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path+target+'_finalmask.fits', overwrite= True)
    print('----> [File saved]:Final mask!!!')



# ========================================================================================================

# ***** Final galaxy segmentation is smoothed by a uniform kernel with size = 10% of the image size. *****
#    global segm_targ
    segm_targ = np.zeros_like(image, dtype=bool)
    segm_targ[(segm_deblend_cold == targ_deb_cold)] = 1 # also should exclude mask
    segm_targ[maskfinal == 1] = 0
    
    boxkernel = Box2DKernel(int(min(nx, ny)/10.))
    boxkernel.normalize()
    convd0 = convolve_fft(segm_targ,boxkernel)
    segmap = convd0 > np.nanmean(convd0.flatten())

    Segm = fits.PrimaryHDU(segmap.astype('float64'))
    targSegm = fits.HDUList([Segm])
    targSegm.writeto(path+target+'_targSegm.fits', overwrite= True)
    print('----> [File saved]:Target segmentation !!!')

# ***** Galaxy target cold segmentation, used to calcalate the mask fraction overlap with the target within 2 Rpet. *****
#    global segm_targ
    segm_targ_all = np.zeros_like(image, dtype=bool)
    segm_targ_all[(segm_cold == targ_cold)] = 1 # also should exclude mask
    segmap_all = segm_targ_all

    Segm_all = fits.PrimaryHDU(segmap_all.astype('float64'))
    targSegm_all = fits.HDUList([Segm_all])
    targSegm_all.writeto(path+target+'_targSegm_all.fits', overwrite= True)
    print('----> [File saved]:Target total segmentation include masked connected pixel!!!')


#   Generate sky mask:
    skymask1 = np.logical_or(segmap,maskfinal)
    skymask_all = np.logical_or(skymask1,skymask)

#
    file_mask = path+target+'_mask.csv'
    open(file_mask, 'w') .close
    
    header=['Name','Sky median','Sky std', 'Mask to total', 'Flag_satu','Flag_merger','Ncomp_galaxy','Ncomp_major','Mag_targ','Mag_pair_min']

    with open(file_mask, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

    data = [target,format(sky_median,'.4f'),format(sky_std,'.4f'), format(mask2total,'.4f'), flag_satu,flag_merger,ncomp_gal,ncomp_pair,format(magCISO,'.4f'),format(mag_comp_min,'.4f')]
    with open(file_mask, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


    return {'final mask': maskfinal, 'target segm': segmap, 'target segm all': segmap_all,'cold deblend segm': segm_deblend_cold,'cold segm': segm_cold,'Cat mask':mask_cat,'Segm mask':mask_segm,'sky mask':skymask_all }














# -----------------------------------------------------------------------------------------------------------------
# Plot segmap using random color:
# -----------------------------------------------------------------------------------------------------------------


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. True or False
        :return: colormap for matplotlib
        """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np
    
    
    
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return
    
    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
                  
        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]
                  
        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
                  
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors,N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]
            
        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]
          
        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))
        
        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)
        
        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional',ticks=None,boundaries=bounds, format='%1i', orientation=u'horizontal')
    
    return random_colormap
