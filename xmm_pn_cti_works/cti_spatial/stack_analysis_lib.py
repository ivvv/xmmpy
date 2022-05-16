# %% [markdown]
# # Library of defs for analysis of a stacked results
# 
# ## Modification history:
# * First version: 27 Oct 2021, Ivan Valtchanov, XMM SOC (SCO-04)
# * 03 Nov 2021: added kernel density estimate instead of binning to a spectrum. Not sure if useful but it gives a better idea of the distribution.
# * 11 Nov 2021: added full RAWX,RAWY processing and saving the results per CCD in a 64x200 array with the offset to 8.04 keV
# * 25 Mar 2022: processing the stack results produced with EPN_CTI_0055/0056.CCF
# * 05 Apr 2022: processing the stack results produced with EPN_CTI_0055/0056.CCF and events binned in 500 revolutions
# * 06 Apr 2022: adapted to run on the grid for each  stacked file
#  
# %%
import os
import time

from astropy.table import Table
from astropy.io import fits
from astropy import stats

import numpy as np
from lmfit.models import GaussianModel, PolynomialModel
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use(['seaborn-colorblind','~/presentation.mplstyle'])

# %%
def fit_single_line(xin,yin,line_c=8.04,use_weights=True,conf=True):
    ''' 
    PURPOSE:
        Fit a model of Gaussian line on a polynomial(2) continuum
    
    INPUTS:
        xin - array, 
            the energy channel (in keV)
        yin - array,
            the counts in channel
        line_c - float,
            is the initial energy of the line in same units as xin
        conf - bool,
            if confidence interval for the parameters is to be calculated
            
    OUTPUTS:
        a tuple of the full fit output class and confidence intervals (if asked)
    
    NOTES:
        * The Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
        * The fit is performed with LMFIT
        
    '''    
    #
    poly_mod = PolynomialModel(2,prefix='poly_')
    pars = poly_mod.guess(yin, x=xin)
    #
    gauss1  = GaussianModel(prefix="g1_")
    pars.update(gauss1.make_params())
    pars['g1_center'].set(line_c,min=line_c-0.25,max=line_c+0.25)
    pars['g1_sigma'].set(0.1,min=0.08,max=0.250)
    #
    mod = poly_mod + gauss1
    #
    if (use_weights):
        yerr = np.sqrt(yin)
        w = np.divide(1.0,yerr,where=yerr != 0)
        try:
            out = mod.fit(yin, pars, x=xin, weights=w,nan_policy='omit')
        except:
            return None
    else:
        try:
            out = mod.fit(yin, pars, x=xin,nan_policy='omit')
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
    return (out,ci_out)

#%
def fit_cu_line(xin,yin,line_c=8.04,use_weights=True):
    ''' 
    PURPOSE:
      Fit the Cu Ka line (8.04 keV), the model is a polynomial(2) + a Gaussian line.
    
    INPUTS:
      xin is the energy channel (in keV)
      yin is the counts
      line_c is the initial energy of the line (in keV)
    
    OUTPUTS:
     a tuple of the full fit output class and the results line in ascii.
    
    NOTES:
      the Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
    '''    
    i1max = np.argmax(yin)
    y1max = yin[i1max]
    #
    poly_mod = PolynomialModel(1,prefix='poly_')
    pars = poly_mod.guess(yin, x=xin)
    #
    pname = 'cuka'
    gauss1  = GaussianModel(prefix=f"{pname}_")
    pars.update(gauss1.make_params())
    pars[f'{pname}_center'].set(line_c,min=line_c-0.25,max=line_c+0.25)
    pars[f'{pname}_sigma'].set(0.1,min=0.08,max=0.250)
    #pars[f'{pname}_amplitude'].set(y1max,min=1.0,max=y1max)
    #
    mod = poly_mod + gauss1
    #init = mod.eval(pars, x=x)
    #out = mod.fit(yin, pars, x=xin, weights=1.0/np.sqrt(yin))
    if (use_weights):
        yerr = np.sqrt(yin)
        w = np.divide(1.0,yerr,where=yerr != 0)
        try:
            out = mod.fit(yin, pars, x=xin, weights=w,nan_policy='omit')
        except:
            return None
    else:
        try:
            out = mod.fit(yin, pars, x=xin,nan_policy='omit')
        except:
            return None
    #
    # confidence intervals on parameters, if needed
    #
    #ci_out = out.conf_interval()
    #print (ci_out['cuka_center'])
    #
    #cen = out.params['g1_center'].value
    #cen_err = out.params['g1_center'].stderr
    #fwhm = out.params['g1_fwhm'].value
    #fwhm_err = out.params['g1_fwhm'].stderr
    #chi2 = out.chisqr
    #df = len(xin)
    #try:
    #    results  = f"{cen:.3f},{cen_err:.3f},{fwhm:.5f},{fwhm_err:.5f},{chi2:.3f},{df}"
    #except:
    #    results = None
    #
    return out

#%%
def fit_mn_line(xin,yin,line_c=5.8988,use_weights=True):
    ''' 
    PURPOSE:
      Fit the Mn Ka line (5.8988 keV), the model is a polynomial(2) + a Gaussian line.
    
    INPUTS:
      xin is the energy channel (in keV)
      yin is the counts
      line_c is the initial energy of the line (in keV)
    
    OUTPUTS:
     a tuple of the full fit output class and the results line in ascii.
    
    NOTES:
      the Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
    '''    
    i1max = np.argmax(yin)
    y1max = yin[i1max]
    #
    poly_mod = PolynomialModel(1,prefix='poly_')
    pars = poly_mod.guess(yin, x=xin)
    #
    pname = 'mnka'
    gauss1  = GaussianModel(prefix=f"{pname}_")
    pars.update(gauss1.make_params())
    pars[f'{pname}_center'].set(line_c,min=line_c-0.25,max=line_c+0.25)
    pars[f'{pname}_sigma'].set(0.1,min=0.08,max=0.250)
    #pars[f'{pname}_amplitude'].set(y1max,min=1.0,max=y1max)
    #
    mod = poly_mod + gauss1
    #init = mod.eval(pars, x=x)
    #out = mod.fit(yin, pars, x=xin, weights=1.0/np.sqrt(yin))
    if (use_weights):
        yerr = np.sqrt(yin)
        w = np.divide(1.0,yerr,where=yerr != 0)
        try:
            out = mod.fit(yin, pars, x=xin, weights=w,nan_policy='omit')
        except:
            return None
    else:
        try:
            out = mod.fit(yin, pars, x=xin,nan_policy='omit')
        except:
            return None
    return out


# %%
def fit_cu_region(xin,yin,use_weights=True):
    '''
    PURPOSE:
      Fit 4 lines in the  Cu-Ka region, Gaussian models for Cu-Ka (8.04 keV), Ni-Ka (7.47 keV), Zn-Ka (8.63 keV), Cu-Kb (8.9 keV) 
      and a polynomial(2) continuum
    INPUTS:
      xin is the energy channel (in keV)
      yin is the counts
    
    OUTPUTS:
     a tuple of the full fit output class and the results line in ascii.
    
    NOTES:
      the Gaussian sigma of the line is only allowed within a certain range: 80 to 250 eV
        
    '''
    i1max = np.argmax(yin)
    y1max = yin[i1max]
    #
    poly_mod = PolynomialModel(1,prefix='poly_')
    pars = poly_mod.guess(yin, x=xin)
    #
    lines = [7.47,8.04,8.63,8.90] # keV
    prfx = ['nika','cuka','znka','cukb']
    mod = poly_mod
    for i,j in zip(lines,prfx):
        gauss = GaussianModel(prefix=f'{j}_')
        pars.update(gauss.make_params())
        pars[f'{j}_center'].set(i,min=i-0.25,max=i+0.25)
        pars[f'{j}_sigma'].set(0.1,min=0.06,max=0.250)
        mod += gauss
    if (use_weights):
        yerr = np.sqrt(yin)
        w = np.divide(1.0,yerr,where=yerr != 0)
        try:
            out = mod.fit(yin, pars, x=xin, weights=w,nan_policy='omit')
        except:
            return None
    else:
        try:
            out = mod.fit(yin, pars, x=xin,nan_policy='omit')
        except:
            return None
    #
    return out
# %%
def run_fit(table,ccd,rawy=[20,200],bin_size=0.1,plot_it=True):
    #
    # Run the fit on the stacked spectra, the spectra are binned with bin_size
    #
    # bin_size is in keV
    #
    if (not isinstance(table,Table)):
        print (f'Input is not an astropy.table.Table, your input is {table.__class__}')
        return None
    if (ccd not in np.arange(1,13,1)):
        print (f'CCD not in [1,12], your input is {ccd}')
        return None
    xmask = table['CCDNR'] == ccd
    #
    tx = table[xmask]
    #
    bins = np.arange(6.0,10.0,bin_size)
    # regular grid for the total best-fit model, just for plotting
    xgrid = np.linspace(6.0,10.0,200)
    mbin = bins[0:-1] + bin_size/2.0
    #
    # now grouping per RAWX
    #
    grpx = tx.group_by('RAWX')
    #
    # list of lines to fit
    lines = [7.47,8.04,8.63,8.90] # keV
    #
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(15,10),sharex=True)
    #
    for grp in grpx.groups:
        rawx = grp['RAWX'][0]
        #if (grp['RAWX'][0] != 32):
        #№    continue
        #
        iq = np.where((grp['RAWY'] < rawy[1]) & (grp['RAWY'] >= rawy[0]))[0]
        yy = np.sort(grp['PI'][iq])/1000.0 # in keV
        nevents = len(iq)
        #
        # bin to form a spectrum
        #
        hist, bin_edges = np.histogram(yy,bins=bins,density=False)
        # Poisson error
        hist_err = np.sqrt(hist)
        #
        # fit with LMFIT
        #
        out = fit_cu_region(mbin,hist)
        # total model on a regular grid
        total = out.eval(x=xgrid)
        if (plot_it):
            ax[0].step(mbin,hist,where='mid',label=f'histo bin={bin_size} eV',zorder=0)
            ax[0].errorbar(mbin,hist,yerr=hist_err,fmt='o',zorder=1)
            # plot the rug if less than 1000 events
            if (nevents < 1000):
                ax[0].plot(yy, np.full_like(yy, -0.1), '|k', markeredgewidth=1,label='data')
            # now fit with LMFIT
            ax[0].plot(xgrid,total,lw=3,zorder=3,label='Total model',color='tab:olive')
            # indicate the position of the fitted 4 lines
            for line in lines:
                ax[0].axvline(line,linestyle='dashed',color='pink',lw=2,zorder=0)
            print ('Fit to histo:\n',out)
            ax[0].legend()
            ax[0].set_ylabel(f'Counts')
            ax[0].grid()
            #
            # residuals
            #
            ax[1].step(mbin,out.residual,label='Histogram',where='mid')
            ax[1].grid()
            ax[1].set_xlabel('PI (keV)')
            ax[1].set_ylabel('Residual')
            ax[1].axhline(0.0,linestyle='dashed',lw=3,color='black')
            ax[0].set_title(f'CCD: {ccd}, RAWX: {rawx}, RAWY: {rawy}, n={nevents}')
        break
    return out

# %%
def run_fit_subsets(table,ccd,rawx=[1],rawy=[20,200],eranges=[7.0,10.0],bin_size=0.01,use_weights=True,model='single',
    use_column='PI',plot_it=True):
    #
    # Run the fit on the stacked spectra, the spectra are binned with bin_size, bin_size is in keV
    # for fixed RAWX, if rawx='all' then will run on all RAWX from 1 to 64.
    # will only use eranges and will bin
    #
    if (not isinstance(table,Table)):
        print (f'Input is not an astropy.table.Table, your input is {table.__class__}')
        return None
    if (ccd not in np.arange(1,13,1)):
        print (f'CCD not in [1,12], your input is {ccd}')
        return None
    #
    if (not (use_column in ['PI','PI_CORR'])):
        print ('Only columns PI or PI_CORR can be used')
        return None
    xmask = table['CCDNR'] == ccd
    #
    tx = table[xmask]
    #
    if (model == 'mnka'):
        eranges = [5.5,6.5]
    bins = np.arange(eranges[0],eranges[1],bin_size)
    # regular grid for the total best-fit model, just for plotting
    xgrid = np.linspace(eranges[0],eranges[1],200)
    #
    # now grouping per RAWX
    #
    grpx = tx.group_by('RAWX')
    #
    # list of lines to show
    lines = [5.8988, 6.04, 7.47,8.04,8.63,8.90] # keV
    #
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(15,10),sharex=True)
    
    results = np.full((64,20),np.nan,dtype=np.single)
    #
    for grp in tqdm(grpx.groups,desc='Process RAWX'):
        irawx = grp['RAWX'][0]
        if (irawx not in rawx):
            continue
        #
        iq = np.where((grp['RAWY'] < rawy[1]) & (grp['RAWY'] >= rawy[0]))[0]
        yy = np.sort(grp[use_column][iq])/1000.0 # in keV
        #
        # check the number of events in Cu-Ka compared to same range nearby (higher energy)
        #
        i1 = np.where((yy <= 8.2) & (yy > 7.8))[0]
        i2 = np.where((yy <= 8.6) & (yy > 8.2))[0]
        #
        check_on = False
        if ((len(i1) < 10) or (len(i1) < 1.2*len(i2))):
            print (f'Skipping RAWX {irawx}, not enough counts above background')
            continue
        else:
            check_on = True
        #
        nevents = len(iq)
        #
        # bin to form a spectrum
        #
        hist, bin_edges = np.histogram(yy,bins=bins,density=False)
        hist_err = np.sqrt(hist)
        xmid = bin_edges[0:-1] + bin_size/2.0
        #
        # now fit with LMFIT
        #
        if (model == '4lines'):
            out = fit_cu_region(xmid,hist,use_weights=use_weights)
        else:
            out = fit_cu_line(xmid,hist,use_weights=use_weights)
        #
        print (f"Centroid: {out.params['cuka_center'].value} +/- {out.params['cuka_center'].stderr}")
        print (f"Reduced chi-sqr: {out.redchi}")
        # total model on a regular grid
        total = out.eval(x=xgrid)
        #
        if (plot_it):
            ax[0].step(xmid,hist,where='mid',color='tab:blue',label=f'histo bin={bin_size} eV',zorder=0)
            if (len(rawx) < 2):
                ax[0].errorbar(xmid,hist,yerr=hist_err,fmt='o',zorder=1,color='tab:blue')
            # plot the rug if less than 1000 events
            if (nevents < 1000):
                ax[0].plot(yy, np.full_like(yy, -0.1), '|k', markeredgewidth=1,label='data')
            ax[0].plot(xgrid,total,lw=3,zorder=3,label='Total model',color='tab:olive')
            # indicate the position of the fitted 4 lines
            for line in lines:
                ax[0].axvline(line,linestyle='dashed',color='pink',lw=2,zorder=0)
            #print ('Fit to histo:\n',results)
            #ax[0].legend()
            ax[0].set_ylabel(f'Counts')
            ax[0].grid()
            #
            # residuals
            #
            ax[1].step(xmid,out.residual,label='Histogram',where='mid',color='tab:grey')
            ax[1].grid()
            ax[1].set_xlabel(f'{use_column} (keV)')
            ax[1].set_ylabel('Residual')
            ax[1].axhline(0.0,linestyle='dashed',lw=3,color='black')
            ax[0].set_title(f'CCD: {ccd}, RAWY: {rawy}')
            plt.show()
    if (not check_on):
        return None
    return out

# %% [markdown]
# ## Fit for all ranges
# 
# The input table with stacked events for an input CCD will be:
# 
# 1. Will group the table results for CCD in `RAWX`: there wil lbe 64 groups for each `RAWX`
# 2. Will bin in `RAWY` direction in 20 pixels regions from 1 to 200, setting the first 12 pixels to 0 as they are not used. Thus the very first bin in `RAWY` will be [12,40] instead of [1,20].
# 3. Events in each group will be binned with `bin_size` keV to build a spectrum.
# 4. The spectrum will be fit for the Cu K$\alpha$ line, `model='single'` or with 3 additional lines, `model='4lines`.
# 5. The result will be an array of 64x10 pixels with the best-fit line energy
# 6. The result will be expanded to an array with dimensions 64x200 to allow visualisation and direct access to each (`RAWX`,`RAWY`)
# 
# Notes: 
# * if the number of counts in [7.8,8.2] keV are equal or smaller than 1.2 times the number of counts in [8.2,8.6] keV, then we'll skip the fit as the Cu K$\alpha$ line is not really above the background.
# * 

# %%
def run_fit_all(table,ccd,bin_size=0.01,eranges=[7.0,10.0],model='single',use_column='PI',verbose=False):
    #
    # Run the fit on the stacked spectra, the spectra are binned with bin_size, bin_size is in keV, default 5 eV
    # for fixed RAWX, if rawx='all' then will run on all RAWX from 1 to 64.
    #
    # model can be '4line' or 'single'
    #
    if (not isinstance(table,Table)):
        print (f'Input is not an astropy.table.Table, your input is {table.__class__}')
        return None
    if (ccd not in np.arange(1,13,1)):
        print (f'CCD not in [1,12], your input is {ccd}')
        return None
    #
    if (not (use_column in ['PI','PI_CORR'])):
        print ('Only columns PI or PI_CORR can be used')
        return None
    #
    xmask = table['CCDNR'] == ccd
    #
    tx = table[xmask]
    #
    bins = np.arange(eranges[0],eranges[1],bin_size)
    #
    # now grouping per RAWX
    #
    grpx = tx.group_by('RAWX')
    rawx = np.arange(1,65,1)
    #
    rawy_array = np.arange(20,200,20)
    # below RAWY=12 no data
    rawy_array[0] = 12
    #
    # create the output arrays, fill with NaNs
    #
    results = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    results_err = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    results_redchi = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    #
    # Loop on RAWX from 1 to 64
    #
    for grp in tqdm(grpx.groups,desc='Processing RAWX'):
        irawx = grp['RAWX'][0]
        ix = irawx - 1
        if (irawx not in rawx):
            continue
        #
        # Loop on RAWY, from 12 to 200 with step 20 (9 bins)
        # Note, there are no events at RAWY < 12
        #
        for iy,irawy in enumerate(rawy_array):
            rawy_start = irawy
            rawy_end = irawy + 20
            iq = np.where((grp['RAWY'] < rawy_end) & (grp['RAWY'] >= rawy_start))[0]
            #
            # skip regions with no data
            #
            if (ccd in [4,10]):
                if ((irawx <= 20) and (rawy_start >= 140)):
                    continue
                if ((irawx > 20) and (rawy_start >= 120)):
                    continue
            #
            if (ccd in [1,7]):
                if ((irawx <= 40) and (rawy_start >= 120)):
                    continue
                if ((irawx > 40) and (rawy_start >= 140)):
                    continue
            #
            yy = np.sort(grp[use_column][iq])/1000.0 # in keV
            # #
            # # check the number of events in Cu-Ka compared to same range nearby (higher energy)
            # #
            i1 = np.where((yy <= 8.2) & (yy > 7.8))[0]
            i2 = np.where((yy <= 8.6) & (yy > 8.2))[0]
            # #
            check_on = False
            if ((len(i1) <= 10) or (len(i1) < 1.2*len(i2))):
                if (verbose):
                    print (f'Skipping RAWX {irawx}, RAWY ({rawy_start},{rawy_end}): not enough counts above background')
                continue
            else:
                 check_on = True
            # #
            # nevents = len(iq)
            #
            # bin to form a spectrum
            #
            hist, bin_edges = np.histogram(yy,bins=bins,density=False)
            mbin = bin_edges[0:-1] + bin_size/2.0
            #
            # now fit with LMFIT
            #
            if (model == '4lines'):
                out = fit_cu_region(mbin,hist,use_weights=True)
                if (out is None):
                    continue
            elif (model == 'single'):
                out = fit_cu_line(mbin,hist,use_weights=True)
                if (out is None):
                    continue
            else:
                print (f'No such model {model} for the 8 keV region')
                return None
            #
            # Saving the result (best fit Cu-Ka)
            results[ix,iy] = out.params['cuka_center'].value
            # error on best-fit energy
            results_err[ix,iy] = out.params['cuka_center'].stderr
            # reduced ChiSqr
            results_redchi[ix,iy] = out.redchi
            #
    #
    # now make it a full RAW CCD with residuals in eV
    #
    yrr = np.full((64,200),np.nan,dtype=np.single)
    yrr_err = np.full((64,200),np.nan,dtype=np.single)
    yrr_redchi = np.full((64,200),np.nan,dtype=np.single)
    #
    for jx in np.arange(64):
        for jy in np.arange(9):
            if (jy == 0):
                i0 = 12
                i1 = 40
            else:
                i0 = jy*20 + 20
                i1 = i0 + 20
            yrr[jx,i0:i1] = 1000*(results[jx,jy] - 8.04)
            yrr_err[jx,i0:i1] = 1000*results_err[jx,jy]
            yrr_redchi[jx,i0:i1] = results_redchi[jx,jy]
    yrr[:,:12] = np.nan
    yrr_err[:,:12] = np.nan
    yrr_redchi[:,:12] = np.nan
    #
    return (yrr,yrr_err,yrr_redchi)

#%%
def plot_results(filename,ccdnr=1,pngfile=None,panels='residual'):
    #
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
def calc_results(start_rev,ccdnr=1,mode='FF',pngfile=None, plot_it=True, stacks_dir=os.getcwd()):
    #
    # compare the CTI distribution of copper (8 keV) before and after the correction
    #
    step = 500
    file0 = f'{stacks_dir}/{mode}_stacked_{start_rev:04}_{start_rev+step-1:04}_results.fits.gz'
    file1 = f'{stacks_dir}/{mode}_stacked_{start_rev:04}_{start_rev+step-1:04}_results_corr.fits.gz'
    if (not (os.path.isfile(file0) and os.path.isfile(file1))):
        raise FileNotFoundError
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
        cbar.set_label('E - 8040 (eV)')
        #ax.set_xticks(rawy_array)
        ax[0].set_xlabel('RAWY')
        #ax.set_ylabel('RAWX')
        ax[0].set_title(f'{mode}, CCD: {ccdnr}, revs in [{start_rev},{start_rev+step-1}]\n mean={xstat0[0]:.1f} eV, st.dev.={xstat0[2]:.1f} eV (3-$\sigma$ clipped)')
        #plt.title(fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)',ha='right',fontsize=16)
        im = ax[1].imshow(resid1,vmin=-100, vmax=100.0,origin='lower',cmap=plt.get_cmap('bwr'))
        div = make_axes_locatable(ax[1])
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('E - 8040 (eV)')
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
