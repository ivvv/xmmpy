

import os
import glob
import numpy as np

from astropy.io import fits
from astropy import stats

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',AstropyWarning)

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse

plt.style.use(['seaborn-colorblind','~/presentation.mplstyle'])

#%%
def compare_before_after(start_rev,ccdnr=1,mode='FF',pngfile=None, plot_it=True, res_dir=os.getcwd()):
    #
    # compare the CTI distribution of copper (8 keV) before and after the correction
    #
    step = 500
    # find the pairs of files
    ffx = glob.glob(f'{res_dir}/{mode}_stacked_{start_rev:04}_{start_rev+step-1:04}*.fits.gz')
    if (len(ffx) != 2):
        print ('Cannot identify pair of files')
        return None
    for ifile in ffx:
        if ('_corr' in ifile):
            file1 = ifile
        else:
            file0 = ifile
    #
    if (not (os.path.isfile(file0) and os.path.isfile(file1))):
        raise FileNotFoundError
    #
    # identify the line by the filename
    #
    if ('mnka' in file0):
        xline = r'Mn K$\alpha$'
    elif ('cuka' in file0):
        xline = r'Cu K$\alpha$'
    else:
        xline = 'Unknown'
    #
    with fits.open(file0) as hdu0, fits.open(file1) as hdu1:
        resid0 = hdu0['RESIDUALS'].data[ccdnr-1,:,:]
        resid1 = hdu1['RESIDUALS'].data[ccdnr-1,:,:]
    #
    # calculate sigma_clipped statistics
    #
    xstat0 = stats.sigma_clipped_stats(resid0, sigma=3, maxiters=3)
    xstat1 = stats.sigma_clipped_stats(resid1, sigma=3, maxiters=3)
    #
    if (plot_it):
        rawy_array = np.arange(20,200,20)
        #
        fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,10),sharex=True)
        #
        im = ax[0].imshow(resid0,vmin=-100, vmax=100.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[0])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Residual (eV)')
        #ax.set_xticks(rawy_array)
        ax[0].set_xlabel('RAWY')
        ax[0].set_ylabel('RAWX')
        ax[0].set_title(f'{mode}, {xline}, CCD: {ccdnr}, revs in [{start_rev},{start_rev+step-1}]\n mean={xstat0[0]:.1f} eV, st.dev.={xstat0[2]:.1f} eV (3-$\sigma$ clipped)')
        #plt.title(fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)',ha='right',fontsize=16)
        im = ax[1].imshow(resid1,vmin=-100, vmax=100.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[1])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Residual (eV)')
        #ax.set_xticks(rawy_array)
        ax[1].set_xlabel('RAWY')
        ax[1].set_ylabel('RAWX')
        ax[1].set_title(f'Corrected\n mean={xstat1[0]:.1f} eV, st.dev.={xstat1[2]:.1f} eV (3-$\sigma$ clipped)')
        if (pngfile is not None):
            plt.savefig(pngfile,dpi=100)
            plt.show()
            #time.sleep(10)
            plt.close() 
        else:
            plt.show()
    return (xstat0,xstat1)

#%%
def main():
    # get the arguments
    parser = argparse.ArgumentParser(description='Display spatial CTI offset')
    parser.add_argument('start_rev', type=int,
                        help='The starting revolution of the stack to use')
    parser.add_argument('ccdnr', type=int,
                        help='The CCDNR to use')
    parser.add_argument('-mode', type=str, default='FF',
                        help='The instrument mode to use, can be FF or EFF')
    parser.add_argument('-res_dir', type=str, default=os.getcwd(),
                        help='The folder with the results')
    args = parser.parse_args()
    #
    out = compare_before_after(args.start_rev,args.ccdnr,mode=args.mode,res_dir=args.res_dir)

if __name__ == "__main__":
    main()
