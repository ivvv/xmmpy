import os
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

def plot_spatial_cti(filename,ccdnr=1,pngfile=None,panels='residual'):
    '''
    PURPOSE:
        Show the derived spatial offsets after fitting for a given line

    INPUTS:
        filename - str,
            The input FITS filename with the derived results. Should contain extensions 'RESIDUALS',
            'ERRORS' and 'CHI2_R'. Unsuccessful fits will be nan.
        ccdnr - int,
            The CCDNR to show
        pngfile - str,
            The name of the output PNG file to save to plot to. If None, then no figure will be saved only shown on screen.
        panels - str,
            Which results to show, can be 'all' or 'residual', with 'all' it will show the thee panels with residual, error and 
            reduced chi-square.
    '''
    #
    #
    if (not os.path.isfile(filename)):
        raise FileNotFoundError
    #
    with fits.open(filename) as hdu:
        resid = hdu['RESIDUALS'].data[ccdnr-1,:,:]
        errors = hdu['ERRORS'].data[ccdnr-1,:,:]
        chi2r = hdu['CHI2_R'].data[ccdnr-1,:,:]
        rev_start=hdu[0].header['REV0']
        rev_end=hdu[0].header['REV1']
    #
    # calculate sigma_clipped statistics
    #
    xstat = stats.sigma_clipped_stats(resid, sigma=3, maxiters=3)
    #
    rawy_array = np.arange(20,200,20)
    if (panels == 'all'):
        fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(10,10),sharex=True)
        im = ax[0].imshow(resid,vmin=-100, vmax=100.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[0])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('E - 8040 (eV)')
        ax[0].set_title(f'CCD: {ccdnr}, revs in [{rev_start},{rev_end}]')
        ax[0].set_xlabel(fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)')
        #ax[0].set_xticks(rawy_array)
        #
        im = ax[1].imshow(errors,vmin=0.0, vmax=20.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[1])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Error (eV)')
        #ax[1].set_title(f'CCD: {ccd}')
        #ax[1].set_xticks(rawy_array)
        #
        im = ax[2].imshow(chi2r,vmin=1.0, vmax=2.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[2])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Chi2_r')
        #ax[2].set_title(f'CCD: {ccd}')
        ax[2].set_xticks(rawy_array)
        #
        if (pngfile is not None):
            plt.savefig(pngfile,dpi=100)
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6),sharex=True)
        im = ax.imshow(resid,vmin=-100, vmax=100.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('E - 8040 (eV)')
        ax.set_xticks(rawy_array)
        ax.set_xlabel('RAWY')
        ax.set_ylabel('RAWX')
        ax.set_title(f'CCD: {ccdnr}, revs in [{rev_start},{rev_end}]\n' + fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)')
        #plt.title(fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)',ha='right',fontsize=16)
        if (pngfile is not None):
            plt.savefig(pngfile,dpi=100)
        plt.show()
    return

#%%

def plot_spatial_cti_all(results_file,pngfile=None):
    '''
    PURPOSE:
        Show the derived spatial offsets after fitting for a given line, all CCDs

    INPUTS:
        filename - str,
            The input FITS filename with the derived results. Should contain extensions 'RESIDUALS',
            'ERRORS' and 'CHI2_R'. Unsuccessful fits will be nan.
        pngfile - str,
            The name of the output PNG file to save to plot to. If None, then no figure will be saved only shown on screen.
    '''

    if (not os.path.isfile(results_file)):
        print (f'Input file with results not found')
        return None
    #
    with fits.open(results_file) as hdu:
        resid = hdu['RESIDUALS'].data
        rev_start=hdu[0].header['REV0']
        rev_end=hdu[0].header['REV1']
    #
    geom = {3:(202,0), 2: (202,66), 1: (202,132), 4: (202,198), 5 : (202, 264), 6: (202,330),
        12:(0,0), 11: (0,66), 10: (0,132), 7: (0,198), 8 : (0, 264), 9: (0,330)}
    #
    epic_pn = np.full((404,404),np.nan)
    # the revolution index
    for iccd in np.arange(1,13):
        off = np.transpose(resid[iccd-1, :, :])
        x0 = geom[iccd][0]
        x1 = geom[iccd][1]
        if (iccd <= 6):
            epic_pn[x0:x0+200,x1:x1+64] = np.flip(off,axis=None)
        else:
            epic_pn[x0:x0+200,x1:x1+64] = off
        xstat = stats.sigma_clipped_stats(off, sigma=3, maxiters=3)
        print (f'CCD {iccd}, stats: {xstat}')
    #
    # plotting
    #
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(epic_pn,vmin=-50, vmax=50,origin='lower',cmap=plt.get_cmap('bwr'))
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Offest (eV)')
    ax.set_title(f'Revs in [{rev_start},{rev_end}]')
    # annotate
    for iccd in np.arange(1,13):
        ax.text(20+geom[iccd][1],180+geom[iccd][0],iccd,color='black',size=20)
    #
    if (pngfile is not None):
        plt.savefig(pngfile,dpi=100)
    plt.show()
    return True

# get the arguments
parser = argparse.ArgumentParser(description='Display spatial CTI offset')
parser.add_argument('filename', type=str,
                    help='The FITS filename with the results')
parser.add_argument('ccdnr', type=int,
                    help='The CCDNR to use')
parser.add_argument('-panels', type=str,default='residual',
                    help='The panels to show: can be residual or all')
args = parser.parse_args()
#
#
if (not os.path.isfile(args.filename)):
    print ("Input FITS file not found.")
    raise FileNotFoundError
#
if (args.ccdnr in np.arange(1,13,1)):
    plot_spatial_cti(args.filename,args.ccdnr,pngfile=None,panels=args.panels)
else:
    plot_spatial_cti_all(args.filename,pngfile=None)


