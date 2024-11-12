import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import statmorph
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel,RickerWavelet2DKernel,convolve_fft
from astropy.convolution import Tophat2DKernel
from astropy.io import fits,ascii
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.visualization import LogStretch,SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.stats import SigmaClip
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from photutils import detect_sources
from photutils import detect_threshold
from photutils import deblend_sources
from photutils import source_properties
from photutils.segmentation import make_source_mask
from photutils.background import Background2D, MedianBackground
from matplotlib import patches
import csv
import matplotlib.pyplot as plt
import ThreeLayerMask as TLM
#import Mask4asymm_YangLi as Asymmask
import numpy.ma as ma
import sys
import warnings
warnings.filterwarnings("ignore")



# Change the mask within 2Rpet. -- Yulin 2021.08.14
# Add statmorph into the process. -- Yulin 2021.07.27
# A Hybrid mask based on Sazonova with Yang's bright core correction. -- Yulin 2021.07.27

# Reference: Sazonova et al. 2021
# Input in terminal: python Photutils_hotcoolcold.py path/fitsname ID ra dec Rpet
# Example to run this code:
# python Photutils_hotcoolcold.py '/Users/path/ 155471139761665605_i' 001 113.9762 39.5625 5.6083
# ID: object id 
# ra, dec: float
# Rpet: Pan-starrs Petrisian radius in arcsec. Rpet = 1.3*Rpet_sdss

#fitsname = str(sys.argv[1])

path = str(sys.argv[1])
target = str(sys.argv[2])
ID = str(sys.argv[3])
ra = float(sys.argv[4])
dec = float(sys.argv[5])
Rpet = float(sys.argv[6])

fitsname = path+target
ID_pan = fitsname[-20:]


hduList = fits.open(fitsname+'.fits')
print ('Target: {0}.fits'.format(fitsname))
image = hduList[0].data
header = hduList[0].header
pixelSize = abs(header['CDELT1']) * 3600.
fwhm0=1.1/0.25

start = time.time()


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
#segtarg = np.array(segm_targ,dtype='bool')

# --------------------------------------- Statmorph
gain = 1
n_Rpetbox = 4
size_skybox = round(n_Rpetbox*Rpet/pixelSize)
#print(size_skybox)

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

try:
    source_morphs = statmorph.source_morphology(image_subsky, segm_targ, gain=gain, mask=mask, segtargall = segmap_all,psf=None, skybox_size= size_skybox,cutout_extent = 20, verbose=True, no_fit = True)

    # cutout_extent: cutout_extent (float, optional) – The target fractional size of the data cutout relative to the minimal bounding box containing the source (Rmax). The value must be >= 1.
    # verbose (bool, optional) – If True, this prints various minor warnings (which do not set off the “bad measurement” flag) during the calculations. The default value is False.
    # no_fit: not using sersic index to do the fit.

    morph = source_morphs[0]

    A_cas = morph.asymmetry
    A_out = morph.outer_asymmetry
    A_shape = morph.shape_asymmetry

    A_cas_randsky = morph.asymmetry_randsky[0]
    A_cas_randsky_error = morph.asymmetry_randsky[1]
    A_out_randsky = morph.outer_asymmetry_randsky[0]
    A_out_randsky_error = morph.outer_asymmetry_randsky[1]

    A_skybox = morph._sky_asymmetry
    A_sky_cas = morph.asym_sky_cas
    A_sky_out = morph.asym_sky_outer

    fracmask_cas = morph.frac_mask_cas
    fracmask_out = morph.frac_mask_out
    fracmask_shape = morph.frac_mask_shape
    fracmask_targ = morph.frac_mask_targ

    flag_maskcenter = morph.flag_maskcenter
    flag_stat = morph.flag
    flag_catastrophic = morph.flag_catastrophic
    
    r50 = morph._radius_at_fraction_of_total_cas(0.5)  # pixel
    rpetro_circ = morph.rpetro_circ   # pixel
    rpetro_ellip = morph.rpetro_ellip    # pixel
    dist2center = morph.distance_center_asymmetry    # pixel

    print('Statmorph measurement is done.')

##  Can add it back if we want to plot figure!!!

    if morph.flag_catastrophic == 1:
        print('There is an catastrophic problem in the statmorph measurement. No figure is plot.')
    else:
        from statmorph.utils.image_diagnostics import make_figure
        fig = make_figure(morph)
        file_stat = path+'/'+target+'_stat.pdf'
        #print(file_stat)
        plt.savefig(file_stat,bbox_inches = 'tight')
        print('Statmorph figure saved.')

    flag_errorstat = 0

except AttributeError as error:
    #print("Unexpected error:", sys.exc_info()[0])
    print('[Error]:There is an error in Statmorph running:', error)
    flag_errorstat = 1


# Write
file_stat = path+target+'_stat.csv'
open(file_stat, 'w') .close

header=['Name','A_skybox','A_cas','A_sky_cas','A_cas_randsky','A_cas_randsky_error','A_out','A_sky_out', 'A_out_randsky','A_out_randsky_error','A_shape','R50','Rpetro_circ','Rpetro_ellip','Dist2center', 'Fracmask_cas','Fracmask_out','Fracmask_shape','Fracmask_targ','Flag_maskedcenter','Flag_errorstat','Flag_stat','Flag_catastrophic']

with open(file_stat, 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

data = [target,format(A_skybox,'.4f'),  format(A_cas,'.4f'),format(A_sky_cas,'.4f'), format(A_cas_randsky,'.4f'),format(A_cas_randsky_error,'.4f'), format(A_out_randsky,'.4f'),format(A_out_randsky_error,'.4f'), format(A_out,'.4f'),format(A_sky_out,'.4f'), format(A_shape,'.4f'), format(r50,'.4f'), format(rpetro_circ,'.4f'), format(rpetro_ellip,'.4f'),format(dist2center,'.4f'), format(fracmask_cas,'4f'), format(fracmask_out,'.4f'), format(fracmask_shape,'.4f'), format(fracmask_targ,'.4f'), flag_maskcenter, flag_errorstat, flag_stat, flag_catastrophic]

with open(file_stat, 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(data)


print('Statmorpht Finish Time: %g s.' % (time.time() - start))


hduList.close()

print('------------------------------------------------------')
print(' ')
print(' ')
print(' ')




