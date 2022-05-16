import numpy as np
from astropy.table import Table
from astropy.modeling import models, fitting

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',AstropyWarning)

import matplotlib.pylab as plt
plt.style.use(['seaborn-colorblind','~/Dropbox/Work/Python/presentation.mplstyle'])

SIG2FWHM = 2.0*np.sqrt(2.0*np.log(2.0))

#%%
def fit_single_line(events: Table,line_c: float=8040, erange: list=(5500.0,10000.0),ebin: float=5.0,
    use_column: str = 'PI',use_weights: bool=True, plot_it: bool=False, plot_file: str=None,verbose: bool=True) -> dict:
    ''' 
    PURPOSE:
        Fit a model of Gaussian line on a polynomial(2) continuum
    
    INPUTS:
        events - Table, the input events list
        line_c - float, the initial energy of the line in same units as PI (or PI_CORR) column
        erange - a list with low and high energy limints in eV for fit and for display
        ebin - float, the energy binning in eV, will bin the events from the table, same unit as PI
        conf - bool, if confidence interval for the parameters is to be calculated
            
    OUTPUTS:
        a tuple of the full fit output class and confidence intervals (if asked)
    
    NOTES:
        * The Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
        * The fit is performed with astropy.modeling
        
    '''    
    if ((use_column == 'PI_CORR') and ('PI_CORR' not in events.columns)):
        print ('Input stacked table has no column PI_CORR, will use PI instead')
        use_column = 'PI'
    #
    # now binning
    #
    emin = erange[0]
    emax = erange[1]
    delta_e = emax - emin
    # for fitting, will use 5 eV binning
    nbins_fit = int(delta_e/ebin)
    hist, bin_edges = np.histogram(events[use_column],bins=nbins_fit,range=erange,density=False)
    fmax = hist.max()
    #
    # get the middle of each bin
    xmid = (bin_edges[0:-1] + bin_edges[1:])/2.0
    # exclude the last bin due to some numerical outliers
    xmid = xmid[0:-1]
    hist = hist[0:-1]
    #
    # check if fitting will make sense
    #
    ix1 = (xmid >= (line_c - 50.0)) & (xmid < (line_c + 50))
    cnt1 = np.sum(hist[ix1])
    #
    ix2 = (xmid <= (emin + 50.0)) | (xmid >= (emax - 50))
    cnt2 = np.sum(hist[ix2])
    if (verbose):
        print (f'Total counts in line: {cnt1}')
        print (f'Total in continuum: {cnt2}, ratio {cnt1/cnt2:.2f}')
    #
    #if (cnt1/cnt2 < 1.2):
    #    print ('Not enough counts in line above the continuum, cannot fit.')
    #    return None
    #
    cont = models.Polynomial1D(2)
    # Gaussian 1d with some bounds
    g1 = models.Gaussian1D(amplitude=fmax, mean=line_c, stddev=100.0,
            bounds={'mean': (line_c-100.0, line_c+100.0), 'stddev': (60.0,250.0)})
    #
    xmodel = cont + g1
    #
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    if (use_weights):
        yerr = np.sqrt(hist)
        w = np.divide(1.0,yerr,where=yerr != 0)
    else:
        w = 1.0
    #
    maxit = 100
    fitted = fit(xmodel, xmid, hist,maxiter=maxit,weights=w)
    if (fitted.stds is None):
        if (verbose):
            print ('Fit failed')
        return None
    #
    # it_limit = 1000
    # check = True
    # while (check and maxit < it_limit):
    #     fitted = fit(xmodel, xmid, hist,maxiter=maxit,weights=w)
    #     if (fitted.stds is None):
    #         maxit += 100
    #     else:
    #         check = False
    erun = np.arange(emin,emax,ebin)
    yfit1 = fitted(erun)
    residual = fitted(xmid) - hist
    chisqr_0 = (residual**2)/fitted(xmid)
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    #
    # storing the results in a dict
    #
    fwhm = SIG2FWHM*fitted.stddev_1.value
    if (fitted.stddev_1.std is None):
        fwhm_err = 0.0
    else:
        fwhm_err = SIG2FWHM*fitted.stddev_1.std
    #
    result = {'chisqr_r': chisqr_r, 'line_c': [fitted.mean_1.value,fitted.mean_1.std],
        #'line_sigma': [fitted.stddev_1.value,fitted.stddev_1.std],
        'line_fwhm': [fwhm,fwhm_err]}
    #
    if (verbose):
        print (f'Solved in {maxit} iterations')
        print (f'Chisqr: {chisqr:.1f}, dof: {len(residual)}')
        print (f'Chisqr_reduced: {chisqr_r:.3f}')
        print (result)
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(10,8))
        #
        # per components
        #
        lx = fitted[0](erun)
        g1x = fitted[1](erun)
        #
        ax[0].step(xmid,hist,where='mid',label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed',label='poly(2)')
        ax[0].plot(erun,g1x,lw=1,ls='dashed',label='Gauss')
        ax[0].set_ylabel('Counts')
        ax[0].grid()
        ax[0].legend()
        #
        ax[1].step(xmid,residual,where='mid',label='Data')
        ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel(f'{use_column} (eV)')
        plt.show()
    return result

#%%
def fit_single_line_lmfit(events: Table,line_c: float=8040, erange: list=(5500.0,10000.0),ebin: float=5.0,
    use_column: str = 'PI',use_weights: bool=True, conf: bool=True, plot_it: bool=False, plot_file: str=None,):
    ''' 
    PURPOSE:
        Fit a model of Gaussian line on a polynomial(2) continuum
    
    INPUTS:
        events - Table, the input events list
        line_c - float, the initial energy of the line in same units as PI (or PI_CORR) column
        erange - a list with low and high energy limints in eV for fit and for display
        ebin - float, the energy binning in eV, will bin the events from the table, same unit as PI
        conf - bool, if confidence interval for the parameters is to be calculated
            
    OUTPUTS:
        a tuple of the full fit output class and confidence intervals (if asked)
    
    NOTES:
        * The Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
        * The fit is performed with LMFIT
        
    '''    
    if ((use_column == 'PI_CORR') and ('PI_CORR' not in events.columns)):
        print ('Input stacked table has no column PI_CORR, will use PI instead')
        use_column = 'PI'
    #
    # now binning
    #
    emin = erange[0]
    emax = erange[1]
    delta_e = emax - emin
    # for fitting, will use 5 eV binning
    nbins_fit = int(delta_e/ebin)
    hist, bin_edges = np.histogram(events[use_column],bins=nbins_fit,range=erange,density=False)
    fmax = hist.max()
    #
    # get the middle of each bin
    xmid = (bin_edges[0:-1] + bin_edges[1:])/2.0
    # exclude the last bin due to some numerical outliers
    xmid = xmid[0:-1]
    hist = hist[0:-1]
    #
    poly_mod = PolynomialModel(2,prefix='poly_')
    pars = poly_mod.guess(hist, x=xmid)
    #
    gauss1  = GaussianModel(prefix="g1_")
    pars.update(gauss1.make_params())
    #
    # line can only be within +/- 200 eV of initial guess
    #
    pars['g1_center'].set(line_c,min=line_c-200.0,max=line_c+200.0)
    pars['g1_amplitude'].set(fmax,min=1.0,max=fmax)
    # sigma can only be in this range [80,250] eV
    #
    pars['g1_sigma'].set(100,min=80,max=250)
    #
    mod = poly_mod + gauss1
    #
    if (use_weights):
        yerr = np.sqrt(hist)
        w = np.divide(1.0,yerr,where=yerr != 0)
        try:
            out = mod.fit(hist, pars, x=xmid, weights=w,nan_policy='omit')
        except:
            return None
    else:
        try:
            out = mod.fit(hist, pars, x=xmid,nan_policy='omit')
        except:
            return None
    if (conf):
        #
        # confidence intervals on parameters, if needed
        #
        ci_out = out.conf_interval()
    else:
        ci_out = None
    #
    # plotting
    #
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(10,8),sharex=True)
        #
        # per components
        #
        total = out.eval(x=xmid)
        ax[0].step(xmid,hist,where='mid',color='tab:blue',label=f'histo bin={ebin} eV',zorder=0)
        ax[0].plot(xmid,total,lw=3,zorder=3,label='Total model',color='tab:olive')
        # indicate the position of the fitted line
        ax[0].axvline(line_c,linestyle='dashed',color='pink',lw=2,zorder=0)
        ax[0].set_ylabel(f'Counts')
        ax[0].grid()
        #
        # residuals
        #
        ax[1].step(xmid,out.residual,label='Histogram',where='mid',color='tab:grey')
        ax[1].grid()
        ax[1].set_xlabel(f'{use_column} (eV)')
        ax[1].set_ylabel('Residual')
        ax[1].axhline(0.0,linestyle='dashed',lw=3,color='black')
        plt.show()
        if (plot_file is not None):
            plt.savefig(plot_file,dpi=100)
            plt.show()
            plt.close()
        else:
            plt.show()
    return (out,ci_out)
