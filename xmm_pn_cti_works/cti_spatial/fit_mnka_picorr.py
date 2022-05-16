# %% [markdown]
# # Analysis of a stacked results
# 
# This is a prototype.
#    
# Will use a stacked events lists for all OBS_IDs, corrected for spatial CTI
# 
# 1. Extract spectra for all RAWX columns (read-out columns) in bins of 20 RAWY rows: there will be 64x10 spectra per CCD.
# 2. Fit each of the spectra for the Cu-K$\alpha$ (8 keV) line and store the result for the line energy.Model can be a single line (CuKa) or 
#    a model of 4 lines in the region
# 
# ## Modification history:
# * First version: 27 Oct 2021, Ivan Valtchanov, XMM SOC (SCO-04)
# * 03 Nov 2021: added kernel density estimate instead of binning to a spectrum. Not sure if useful but it gives a better idea of the distribution.
# * 11 Nov 2021: added full RAWX,RAWY processing and saving the results per CCD in a 64x200 array with the offset to 8.04 keV
# * 25 Mar 2022: processing the stack results produced with EPN_CTI_0055/0056.CCF
# * 05 Apr 2022: processing the stack results produced with EPN_CTI_0055/0056.CCF and events binned in 500 revolutions
# * 06 Apr 2022: adapted to run on the grid for each  stacked file, also moved the defs in a library file
# * 13 Apr 2022: adapted to run for with PI_CORR read from the relevant files
# * 10 May 2022: added skip option, if to skip already available results
# * 12 May 2022: adapted to fit Mn Ka line, using the new library of defs
# * 16 May 2022: added input parameter to select PI or PI_CORR for the fit
#
#  
# %%
import os
import sys
import argparse

from astropy.table import Table
from astropy.io import fits
from tqdm import trange

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',AstropyWarning)
  
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import date
home = os.path.expanduser('~')

#sys.path.append(f'{home}/Dropbox/Work/XMM/xmmpy/cti/sanders')
#sys.path.append(f'/xdata/xcaldata/XMM/IVAN/sanders')
#sys.path.append(f'{home}/GitHub/xmmpy/xmm_pn_cti_works/cti_spatial')

from run_fit_for_mnka import run_fit_for_mnka

# %%
home = os.path.expanduser('~')

# %%

# get the arguments
parser = argparse.ArgumentParser(description='Spatial CTI calculations')
parser.add_argument('start_rev', type=int,
                    help='The first revolution to use')
parser.add_argument('-stacks_dir', type=str, default='/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056',
                    help='The folder with stacks')
parser.add_argument('-use_column', type=str, default='PI_CORR',
                    help='Which column to use for the fit, can be PI or PI_CORR')
parser.add_argument('-mode', type=str, default='FF',
                    help='The observation mode to use, can be FF or EFF')
parser.add_argument('--skip', default=False, action='store_true',
                    help='Skip if output file already exists.')
args = parser.parse_args()
#
if (args.stacks_dir is None):
    wdir = os.getcwd()
else:
    wdir = args.stacks_dir
#
if (not os.path.isdir(wdir)):
    print (f'Cannot find the folder with the stacked event lists: {wdir}')
    raise FileNotFoundError
#
results_dir = f'{wdir}/results_picorr_mnka'
if (not os.path.isdir(results_dir)):
    os.mkdir(results_dir)
#
rstep = 500
# store in single-precision float array
# one file per stacked
output = np.full((12,64,200),np.nan,dtype=np.single)
output_err = np.full((12,64,200),np.nan,dtype=np.single)
output_redchi = np.full((12,64,200),np.nan,dtype=np.single)
output_nevts = np.full((12,64,200),0,dtype=int)
#
xmode = args.mode
rev0 = args.start_rev
rev1 = rev0 + rstep - 1
#
if (args.use_column not in ['PI','PI_CORR']):
    print ('Error! Only PI or PI_CORR can be used for the fit.')
    raise RuntimeError
#
if (args.use_column == 'PI_CORR'):
    savefile = f'{results_dir}/{xmode}_stacked_{rev0:04}_{rev1:04}_mnka_corr.fits.gz'
else:
    savefile = f'{results_dir}/{xmode}_stacked_{rev0:04}_{rev1:04}_mnka.fits.gz'
#
print (f'*** Doing revolution {rev0:04} to {rev1:04}, fit for Mn Ka using {args.use_column}')
#
if (os.path.isfile(savefile) and args.skip):
    print (f'Results file {savefile} already exists and skip is {args.skip}. Will skip recalculating it again.')
    sys.exit(0)
#
# Stacks are now per CCD
for j in np.arange(1,13,1):
    # read the stacked file
    sfile = f'{wdir}/corrected/{xmode}_stacked_CCD{j:02}_{rev0:04}_{rev1:04}_0056_corr.fits.gz'
    if (not os.path.isfile(sfile)):
        print (f'No stacked file found: {sfile}')
        raise FileNotFoundError
    t = Table.read(sfile)
    print ('Doing CCD:',j)
    ww = run_fit_for_mnka(t,use_column=args.use_column,verbose=False)
    output[j-1,:,:] = ww[0]
    output_err[j-1,:,:] = ww[1]
    output_redchi[j-1,:,:] = ww[2]
    output_nevts[j-1,:,:] = ww[3]
#
# save to a FITS image
#
hdu0 = fits.PrimaryHDU()
hdu1 = fits.ImageHDU(output, name='RESIDUALS')
hdu2 = fits.ImageHDU(output_err, name='ERRORS')
hdu3 = fits.ImageHDU(output_redchi, name='CHI2_R')
hdu4 = fits.ImageHDU(output_nevts, name='NEVENTS')

hdul = fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
hdu0.header['REV0'] = rev0
hdu0.header['REV1'] = rev1
hdu0.header['MODE'] = xmode
hdu0.header['HISTORY'] = f'Created by Ivan V, using fit_mnka_picorr, {date.today()}'
hdu0.header['COMMENT'] = f'Using events with args.use_column'
hdul.writeto(savefile,overwrite=True)
#
print (f'Results for [{rev0},{rev1}] saved to {savefile}')
print ("*** All done")
#
