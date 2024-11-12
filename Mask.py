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
# z: redshift of target

#fitsname = str(sys.argv[1])

path = str(sys.argv[1])
target = str(sys.argv[2])
ID = str(sys.argv[3])
ra = float(sys.argv[4])
dec = float(sys.argv[5])
Rpet = float(sys.argv[6])
z = float(sys.argv[7])

fitsname = path+target
ID_pan = fitsname[-20:]
#print(ID_pan)

hduList = fits.open(fitsname+'.fits')
print ('Target: {0}.fits'.format(fitsname))
image = hduList[0].data

header = hduList[0].header
pixelSize = abs(header['CDELT1']) * 3600.
exptime = header['EXPTIME'] # sec
wcs = WCS(header)
srcPstXY = wcs.all_world2pix([ra], [dec], 0)
srcXp = srcPstXY[0][0]
srcYp = srcPstXY[1][0]
posi=[srcXp,srcYp]
fwhm0=1.1/0.25


# 定义全局变量：
param=TLM.params(fwhm0,path,fitsname0=target+'.fits',z0 = z,Rpet=Rpet,pixelSize0=pixelSize)  #targposi0=(5650,4750),

start = time.time()

threelayermask = TLM.HotCoolCold(image, posi,snr = 'SNR3' ) #,skystd = sky_std
segm_targ = threelayermask['target segm']
segm_targ_all = threelayermask['target segm all']
segm_cold = threelayermask['cold segm']
segm_deblend_cold = threelayermask['cold deblend segm']
mask_cat = threelayermask['Cat mask']
mask_segm = threelayermask['Segm mask']
finalmask = threelayermask['final mask']
skymask = threelayermask['sky mask']

print('Mask Finish Time: %g s.' % (time.time() - start))


# ------------------------------------------------------------------------------
# Plot check image:
# ------------------------------------------------------------------------------

sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, mask=skymask,maxiters=None)
image_subsky = image-sky_median

img_msk0 = image_subsky.copy()
img_msk0[finalmask==1] = -99
vmin = -sky_std
vmax = np.nanquantile(img_msk0.flatten(),0.9998)

img_msk = image_subsky.copy()
img_msk[finalmask==1] = vmin

plt.figure(figsize=(20,10))
#plt.figure(figsize=(16,16))

plt.subplot(241)
norms = simple_norm(image-sky_median, 'asinh', min_cut = vmin/2, max_cut = vmax)

plt.imshow(image-sky_median, origin='lower', cmap='Greys_r',interpolation='nearest',norm=norms)
ax = plt.gca()
ax.text(0.05,0.95,target,fontsize = 16,color='black',transform=ax.transAxes,verticalalignment='top',
        horizontalalignment='left',bbox={'facecolor':'white', 'edgecolor':'none','alpha':0.6})
plt.title('Data',fontsize=18)

##
plt.subplot(248)
plt.imshow(segm_targ, origin='lower', cmap='Greys_r', interpolation='nearest')
plt.title('Target segmap',fontsize=18)

##
#plt.subplot(248)
#plt.imshow(segm_targ_all, origin='lower', cmap='Greys_r', interpolation='nearest')
#plt.title('Original target segmap',fontsize=18)

#
segm_se = fits.getdata(fitsname+'_Deb1SEGM.fits')
plt.subplot(242)
nlabel = max(segm_se.flatten())
new_cmap = TLM.rand_cmap(nlabel+1, type='bright', first_color_black=True, last_color_black=False, verbose=False)
plt.imshow(segm_se, origin='lower', cmap=new_cmap, vmin=0, vmax=nlabel, interpolation='nearest')
plt.title('Star segmap',fontsize=18)

segm_se = fits.getdata(fitsname+'_Deb2SEGM.fits')
plt.subplot(243)
nlabel = max(segm_se.flatten())
new_cmap = TLM.rand_cmap(nlabel+1, type='bright', first_color_black=True, last_color_black=False, verbose=False)
plt.imshow(segm_se, origin='lower', cmap=new_cmap, vmin=0, vmax=nlabel, interpolation='nearest')
plt.title('Gal<2Rpet segmap',fontsize=18)
#

segm_out = fits.getdata(fitsname+'_Deb3SEGM.fits')
plt.subplot(244)
nlabel = max(segm_out.flatten())
new_cmap = TLM.rand_cmap(nlabel+1, type='bright', first_color_black=True, last_color_black=False, verbose=False)
plt.imshow(segm_out, origin='lower', cmap=new_cmap, vmin=0, vmax=nlabel, interpolation='nearest')
plt.title('Gal>2Rpet+Satustar segmap',fontsize=18)
#

plt.subplot(245)
plt.imshow(img_msk, origin='lower', cmap='Greys_r',interpolation='nearest',norm=norms)

ax=plt.gca()
a=2*Rpet/pixelSize
ellipse = patches.Ellipse(xy=(posi[0],posi[1]),width=2*a,height=2*a,angle=90,color='white',fill=False,alpha=1,linestyle='--')
ax.add_patch(ellipse)
plt.title('Data with final mask',fontsize=18)

#
plt.subplot(246)
maskini=np.zeros_like(image, dtype=bool)
maskini[np.isnan(image)] = 1
maskini[np.isinf(image)] = 1
mask_cat = np.logical_or(mask_cat,maskini)
plt.imshow(mask_cat.astype('float'),origin='lower',interpolation='nearest')
plt.title('gal_in+star_unsa mask',fontsize=18)

#
plt.subplot(247)
#cmap = segm_deblend_cold.make_cmap(seed=223)
plt.imshow(mask_segm, origin='lower', interpolation='nearest')
plt.title('gal_out+star_satu mask',fontsize=18)
# ax2.add_patch(ellipse)


plt.savefig(fitsname+'_mask.pdf',bbox_inches='tight')
print('----> [File saved]:mask pdf!')


print('Plotting Finish Time: %g s.' % (time.time() - start))

hduList.close()


print('------------------------------------------------------')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')
print(' ')

