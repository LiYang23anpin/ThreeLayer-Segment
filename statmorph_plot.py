import numpy as np
#import statmorph
from astropy.io import fits,ascii
from astropy.visualization import simple_norm
from astropy.visualization import LogStretch,SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.patches import Ellipse, Circle,Rectangle
import warnings
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import statmorph
import skimage.transform
warnings.filterwarnings("ignore")

path = './test_stat/'
target = '2360_163311981653160192_i'
fitsname = path+target

hduList = fits.open(fitsname+'.fits')
print ('Target: {0}.fits'.format(fitsname))
image = hduList[0].data
header = hduList[0].header
pixelSize = abs(header['CDELT1']) * 3600.
fwhm0=1.1/0.25
ny, nx = image.shape
#print(ny,nx)

hduList_mask = fits.open(fitsname+'_finalmask.fits')
finalmask = hduList_mask[0].data
mask = np.array(finalmask,dtype='bool')

hduList_segall = fits.open(fitsname+'_targSegm_all.fits')
segall = hduList_segall[0].data
segmap_all = np.array(segall,dtype='bool')

skymask = make_source_mask(image, nsigma=3, npixels=5, dilate_size=11)

skymask1 = np.logical_or(segmap_all,mask)
skymask_all = np.logical_or(skymask1,skymask)

sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, mask=skymask_all,maxiters=None)
image_subsky = image-sky_median

hduList_segtarg = fits.open(fitsname+'_targSegm.fits')
segm_targ = hduList_segtarg[0].data



# --------------------------------------- Statmorph ---------------------------------------#
gain = 1
n_Rpetbox = 4
Rpet = 5.5806517
size_skybox = round(n_Rpetbox*Rpet/pixelSize)

A_cas = -999
A_out = -999
A_shape = -999
A_cas_randsky = -999
A_out_randsky = -999
A_cas_randsky_error = -999
A_out_randsky_error = -999
A_skybox = -999
A_sky_cas = -999
A_sky_out = -999
A_sky_cas = -999
A_sky_out = -999
A_sky_shape = -999

fracmask_cas = -999
fracmask_out = -999
fracmask_shape = -999
fracmask_targ = -999
dist2center = -999

flag_maskcenter = 0
flag_catastrophic = 0
flag_stat = 0
source_morphs = statmorph.source_morphology(image_subsky, segm_targ, gain=gain, mask=mask, segtargall = segmap_all,psf=None, skybox_size= size_skybox,cutout_extent = 20, verbose=True, no_fit = True)
morph = source_morphs[0]



# ----------------------------------- Plot -------------------------------------
plt.style.use("classic")
plt.rc('font',family='Times New Roman')

plt.figure(figsize=(25,6))
plt.subplots_adjust(hspace =0, wspace=0.005)


##################
# Original image #
##################
plt.subplot(141)
image0 = np.float64(morph._cutout_stamp_maskzeroed) 
ny, nx = image0.shape
print(ny,nx)
Lcut=55
image = image0[Lcut:(ny-Lcut),Lcut:(nx-Lcut)]
nycut, nxcut = image.shape
xc, yc = morph._xc_stamp, morph._yc_stamp  
xca, yca = morph._asymmetry_center   # asym. center
theta_vec = np.linspace(0.0, 2.0*np.pi, 200)
finalmask0 = morph._mask
finalmask = finalmask0[Lcut:(ny-Lcut),Lcut:(nx-Lcut)]
img_msk = ma.array(image,mask=finalmask)
image[finalmask==1] = -99
norms1 = simple_norm(image, 'linear', percent=99.999 )

ax=plt.gca()
my_cmap = cm.Greys_r
my_cmap.set_under('k', alpha=0)
plt.imshow(np.clip(image, a_min=0.6, a_max=None),
            origin='lower',
            cmap=my_cmap,
            norm=ImageNormalize(stretch=LogStretch(), clip=False),
            clim=[0.1 * sky_std, None],
            vmin=0.12)


a = morph.rhalf_ellip
b = a / morph.elongation_asymmetry
theta = morph.orientation_asymmetry
xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta))
y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta))

contour_levels = [0.5]
contour_colors = [(1, 1, 1)]
segmap_stamp0 = morph._segmap.data[morph._slice_stamp]
segmap_stamp = segmap_stamp0[Lcut:(ny-Lcut),Lcut:(nx-Lcut)]
Z = np.float64(segmap_stamp == morph.label)
ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5, labels='Target segmentation map')

#plt.legend(loc=4, fontsize=16, facecolor='w', framealpha=0.8, edgecolor='k',ncol=2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.text(14,nycut-40,'(a)',fontsize=34,color='w')

Rect = Rectangle(xy = (90,8), width = 238, height = 68, alpha = 0.76, facecolor='w',edgecolor='k')
ax.add_patch(Rect)
ax.plot([106,136],[55,55],color='w',linewidth=2)
ax.text(144,52,' Target segment',fontsize=26)

ax.scatter(120,28,marker='o',color='k',s=250,zorder=4)
ax.text(144,20,' Masked region',fontsize=26)


######################
#    CAS residual    #
######################
plt.subplot(142)
image_180 = skimage.transform.rotate(image0, 180.0, center=(xca, yca))
image_res0 = image0 - image_180
mask0 = morph._mask_stamp.copy()
mask_180 = skimage.transform.rotate(mask0, 180.0, center=(xca, yca))
mask_180 = mask_180 >= 0.5  # convert back to bool
mask_symmetric = mask0 | mask_180
image_res0 = np.where(~mask_symmetric, image_res0, 0.0)

image_res = image_res0[Lcut:(ny-Lcut),Lcut:(nx-Lcut)]
image_res[finalmask==1] = -99

plt.imshow(np.clip(image_res, a_min=0.6, a_max=None),
            origin='lower',
            cmap=my_cmap,
            norm=ImageNormalize(stretch=LogStretch(), clip=False),
            clim=[0.1 * sky_std, None],
            vmin=0.12)
ax=plt.gca()
my_cmap = cm.Greys_r
my_cmap.set_under('k', alpha=0)
ax.plot(xca-Lcut, yca-Lcut, 'c',marker='o', markersize=7, label='Asymmetry center')
r = 1.5*morph.rpetro_circ
ax.plot(xca + r*np.cos(theta_vec)-Lcut, yca + r*np.sin(theta_vec)-Lcut, 'c', linestyle='--',
        label=r'$1.5*R_{\rm P}$',linewidth=2.5)
text = (r'$A_{\rm CAS} = %.2f$' % (morph.asymmetry_randsky[0]))
ax.text(0.44, 0.85, text, fontsize=32,
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, color='w',weight='demibold' )

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#ax.legend(loc=4, fontsize=16, facecolor='w', framealpha=0.8, edgecolor='k')
ax.text(14,nycut-40,'(b)',fontsize=34,color='w')
plt.xlim(0,336)
plt.ylim(0,336)

Rect = Rectangle(xy = (85,8), width = 243, height = 68, alpha = 0.76, facecolor='w',edgecolor='k')
ax.add_patch(Rect)
plt.scatter(106, 57, marker='o',color='c',edgecolor='k',s=54,zorder=4)
ax.text(126,50,' Asymmetry center',fontsize=26)
ax.plot([102,130],[25,25],color='c',linewidth=2,linestyle='--')
ax.text(126,20,r' 1.5$\,R_{\rm P}$',fontsize=26)


######################
#   Outer residual   #
######################
ax=plt.subplot(143)
norms3 = simple_norm(image_res, 'log', min_cut = 0.12)
plt.imshow(np.clip(image_res, a_min=0.5, a_max=None),
            origin='lower',
            cmap=my_cmap,
            norm=ImageNormalize(stretch=LogStretch(), clip=False),
            clim=[0.1 * sky_std, None],
            vmin=0.12)

a = morph.rhalf_ellip
b = a / morph.elongation_asymmetry
theta = morph.orientation_asymmetry
xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta)) - Lcut
y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta)) - Lcut
ax.plot(x, y, 'r', label=r'$a = R_{\,\rm half}$',linewidth=2.5)

a2 = morph.rmax_ellip
b2 = a2 / morph.elongation_asymmetry
xprime2, yprime2 = a2*np.cos(theta_vec), b2*np.sin(theta_vec)
xm = xca + (xprime2*np.cos(theta) - yprime2*np.sin(theta)) - Lcut
ym = yca + (xprime2*np.sin(theta) + yprime2*np.cos(theta)) - Lcut
ax.plot(xm, ym, 'r', label=r'$a=R_{\,\rm max}$',linestyle='--',linewidth=2.5)

text = (r'$A_{\rm outer} = %.2f$' % (morph.outer_asymmetry_randsky[0],))
ax.text(0.43, 0.85, text, fontsize=32,
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, color='w',weight='demibold')

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend(loc=4, fontsize=26, facecolor='w', framealpha=0.8, edgecolor='k')
ax.text(14,nycut-40,'(c)',fontsize=34,color='w')
plt.xlim(0,336)
plt.ylim(0,336)


##########################
# Shape asymmetry segmap #
##########################
ax=plt.subplot(144)
shape_segm = morph._segmap_shape_asym
shape = shape_segm[Lcut:(ny-Lcut+1),Lcut:(nx-Lcut+1)]
plt.imshow(shape, cmap='Greys_r', origin='lower')
ax.plot(xca-Lcut, yca-Lcut, 'c',marker='o', markersize=7, label='Asymmetry center')

r = morph.rmax_circ
ax.plot(xca + r*np.cos(theta_vec)-Lcut,
        yca + r*np.sin(theta_vec)-Lcut,
        'lime', lw=2.5, label=r'$R_{\rm max,circ}$',linestyle='--')

text = (r'$A_{\rm shape} = %.2f$' % (morph.shape_asymmetry,))
ax.text(0.43, 0.85, text, fontsize=32,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,color='w',weight='demibold')

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.text(20,nycut-40,'(d)',fontsize=34,color='w')
plt.xlim(0,336)
plt.ylim(0,336)
Rect = Rectangle(xy = (85,8), width = 243, height = 68, alpha = 0.76, facecolor='w',edgecolor='k')
ax.add_patch(Rect)
plt.scatter(106, 57, marker='o',color='c',edgecolor='k',s=54,zorder=4)
ax.text(126,50,' Asymmetry center',fontsize=26)
ax.plot([100,128],[25,25],color='lime',linewidth=2,linestyle='--')
ax.text(126,20,r' $R_{\rm max}$',fontsize=26)

## plt.savefig('./Figure/statmorph_result_2360.pdf',bbox_inches='tight')