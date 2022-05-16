import numpy as np

from astropy.table import Table
from astropy.modeling import models, fitting

import matplotlib.pylab as plt
plt.style.use(['seaborn-colorblind','~/Dropbox/Work/Python/presentation.mplstyle'])


def fit6lines(events: Table,erange: list=(5600.0,10000.0),ebin: float=5.0,use_column: str = 'PI', 
        plot_it: bool=False, plot_file: str=None, verbose: bool=False,labels: bool=True) -> dict:
    '''
    Purpose:
        Fitting a 6 lines model for instrumental lines around 8 keV
        Mn Ka (5898 eV) Mn Kb (6490 eV), Ni Ka (7470 eV), Cu Ka (8038 eV), Zn Ka (8630 eV) and Cu Kb (8900 eV)

    Inputs:
        events - an astropy Table from the stacking results, must have column PI
        erange - a list with low and high energy limints in eV for fit and for display
        ebin - float, the energy binning in eV, will bin the events from the table
        use_column - str, which energy column to use, can be 'PI' or 'PI_CORR' for CTI corrected
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    Notes:
        Uses astropy.modeling for the fit
    '''
    if ((use_column == 'PI_CORR') and ('PI_CORR' not in events.columns)):
        print ('Input stacked table has no column PI_CORR, will use PI instead')
        use_column = 'PI'
    #
    line_names = {'mnka': r'Mn K$\alpha$','mnkb': r'Mn K$\beta$','nika': r'Ni K$\alpha$', 
                  'cuka': r'Cu K$\alpha$', 'znka': r'Zn K$\alpha$', 'cukb': r'Cu K$\beta$'}
    lines0 = {'mnka': 5898.8,'mnkb': 6490.0,'nika': 7470.0, 'cuka': 8038.0, 'znka': 8630.0, 'cukb': 8900.0}
    #
    emin = erange[0]
    emax = erange[1]
    delta_e = emax - emin
    # for fitting, will use 5 eV binning
    nbins_fit = int(delta_e/ebin)
    hist1, bin_edges1 = np.histogram(events[use_column],bins=nbins_fit,range=erange,density=False)
    fmax = hist1.max()
    # fitting model, linear + Gaussian (Ni Ka) + Gaussian (Cu Ka) + Gaussian (Zn Ka) + Gaussian (Cu Kb)
    #l = models.Linear1D()
    cont = models.Polynomial1D(2)
    g0 = models.Gaussian1D(amplitude=fmax/2.0, mean=5900, stddev=100.0)
    g1 = models.Gaussian1D(amplitude=fmax/2.0, mean=6490, stddev=100.0)
    g2 = models.Gaussian1D(amplitude=fmax/2.0, mean=7470, stddev=100.0)
    g3 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    g4 = models.Gaussian1D(amplitude=fmax/3.0, mean=8630.0, stddev=100.0)
    g5 = models.Gaussian1D(amplitude=fmax/4.0, mean=8900.0, stddev=100.0)
    # get the middle of each bin
    xmid = (bin_edges1[0:-1] + bin_edges1[1:])/2.0
    # exclude the last bin due to some numerical outliers
    xmid = xmid[0:-1]
    hist1 = hist1[0:-1]
    #
    xmodel = cont + g0 + g1 + g2 + g3 + g4 + g5
    #
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    maxit = 100
    check = True
    while (check and maxit <= 2000):
        fitted = fit(xmodel, xmid, hist1,maxiter=maxit,weights=1.0/np.sqrt(hist1))
        if (fitted.stds is None):
            maxit += 100
        else:
            check = False
    #
    #fitted = fit(xmodel, xmid, hist1,maxiter=200,weights=1.0/np.sqrt(hist1))
    #fitted = fit(xmodel, xmid, hist1,maxiter=200)
    if (verbose):
        print (f'Solved in {maxit} iterations')
        print (fitted.stds)
    #
    erun = np.arange(emin,emax,ebin)
    yfit1 = fitted(erun)
    residual = fitted(xmid) - hist1
    ratio = hist1/fitted(xmid)
    chisqr_0 = (residual**2)/fitted(xmid)
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    lines = {'chisqr_r': chisqr_r, 
             'mnka': [fitted.mean_1.value,fitted.mean_1.std], 
             'mnkb': [fitted.mean_2.value,fitted.mean_2.std], 
             'nika': [fitted.mean_3.value,fitted.mean_3.std],
             'cuka': [fitted.mean_4.value,fitted.mean_4.std],
             'znka': [fitted.mean_5.value,fitted.mean_5.std], 
             'cukb': [fitted.mean_6.value,fitted.mean_6.std]}
    peaks = {'mnka': fitted.amplitude_1.value, 
             'mnkb': fitted.amplitude_2.value, 
             'nika': fitted.amplitude_3.value,
             'cuka': fitted.amplitude_4.value,
             'znka': fitted.amplitude_5.value, 
             'cukb': fitted.amplitude_6.value}
    #
    if (verbose):
        print (f'Chisqr: {chisqr:.1f}, dof: {len(residual)}')
        print (f'Chisqr_reduced: {chisqr_r:.3f}')
        print (lines)
    #
    # return results
    #
    #
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(10,8),sharex=True)
        #
        # per components
        #
        lx = fitted[0](erun)
        g0x = fitted[1](erun)
        g1x = fitted[2](erun)
        g2x = fitted[3](erun)
        g3x = fitted[4](erun)
        g4x = fitted[5](erun)
        g5x = fitted[6](erun)
        #
        #ax[0].hist(events['PI'],bins=nbins_fit,range=erange,histtype='step',label='Data',density=False)
        ax[0].plot(xmid,hist1,label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g0x,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].plot(erun,g2x,lw=1,ls='dashed')
        ax[0].plot(erun,g3x,lw=1,ls='dashed')
        ax[0].plot(erun,g4x,lw=1,ls='dashed')
        ax[0].plot(erun,g5x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].set_title(f'6-lines model, using {use_column}')
        ax[0].grid()
        ax[0].legend()
        if (labels):
            for k in lines0.keys():
                ax[0].text(lines0[k],peaks[k],line_names[k],rotation='vertical')
        #
        ax[1].plot(xmid,ratio,label='Data/Model')
        ax[1].axhline(1.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].set_ylabel('Data/Model')
        ax[1].set_ylim([0.9,1.1])
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        #ax[1].plot(xmid,residual,label='Data')
        #ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        #ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        if (plot_file is not None):
            plt.savefig(plot_file,dpi=100)
            plt.show()
            plt.close()
        else:
            plt.show()
    return lines
