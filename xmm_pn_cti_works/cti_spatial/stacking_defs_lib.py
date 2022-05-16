#
# Library of defs for energy scale and long-term CTI analysis using stacked event lists data
#

import os
import glob

import numpy as np
import pandas as pd
import pickle

from astropy.table import Table, vstack
from astropy.modeling import models, fitting

import matplotlib.pylab as plt
plt.style.use(['seaborn-colorblind','~/Dropbox/Work/Python/presentation.mplstyle'])

home = os.path.expanduser('~')

def bin_events(events: Table,erange: list[int,int]=(5500,10000),ebin:float=5.0) -> Table:
    '''
    Purpose:
        Bin the input events to create a spectrum
    Inputs:
        events - astropy.Table, the input event list as table, must have column 'PI'
        erange - list, with (PI.min,PI.max) ranges in eV to consider for the binning
        ebin - float, the bin size in eV, default as in XMM-SAS is 5 eV
    Output:
        astropy.Table with two columns: 'bin' and 'counts', wheren 'bin' is the binned centre.

    '''
    bin_edges = np.arange(erange[0],erange[1],ebin)
    hist, bin_edges = np.histogram(events['PI'],bins=bin_edges,range=erange,density=False)
    xmid = (bin_edges[0:-1] + bin_edges[1:])/2.0
    # skip the last one
    table = Table([xmid[0:-1],hist[0:-1]],names=['bin','counts'],dtype=[float,int])
    return table

def extract_events(stack_table: Table,ccdnr: int=None, obsid:int=None, verbose:bool=True) -> list[Table,int]:
    '''
    Purpose:
        Extract events from stacked file, for a given CCDNR or OBSID
    Inputs:
        stack_table - Table, the input stacked table
        ccdnr - int, the CCD number, from 1 to 12
        obsid - int, the XMM OBS_ID
        verbose - bool, whether to print verbose information
    Output:
        (Table,nobs): list
        Table with the extracted events
        nobs  the number of unique observations in the output table
    Notes:
        If `ccd` is None, then will extract all CCDs for a given OBSIDs
        If `obsid` is None, then will extract all OBSIDs from the file
        `ccd` and `obsid` cannot be both None
    '''
    if (ccdnr is None and obsid is None):
        print ('Cannot have both ccdnr and obsid as None. Exit')
        return None
    #
    if (not os.path.isfile(stack_table)):
        print (f'Input table {stack_table} not found. Exit')
        return None
    #
    t = Table.read(stack_table)
    if (ccdnr is None):
        result = t[(t['OBSID'] == obsid)]
        if (verbose):
            print (f'Extracted {len(result)} events for OBSID {obsid} and all CCDs')
    elif (obsid is None):
        result = t[(t['CCDNR'] == ccdnr)]
        nobsids = len(np.unique(result['OBSID']))
        if (verbose):
            print (f'Extracted {len(result)} events for CCDNR {ccdnr} and {nobsids} observations.')
    else:
        result = t[(t['CCDNR'] == ccdnr) & (t['OBSID'] == obsid)]
        if (verbose):
            print (f'Extracted {len(result)} events for CCDNR {ccdnr} and OBSID {obsid}')
    return [result, nobsids]

def merge_stacks(ccdnr: int,rev_range: list[int,int],bin_spec: bool=True, mode: str='FF', 
                 stacks_folder: str='.',table_suffix='0055_0056',verbose=True) -> list[Table,int]:
    '''
    Merge events from different stacks for a given mode (FF or EFF)
    
    The stack tables for FF contain events combined for 100 revolutions
    
    Inputs:
        ccdnr - int, the CCD number to extract events
        rev_range - list, events for `ccdnr` from observations in [`rev_range`[0],`rev_range`[1]] will be merged in a table
        bin_spec - bool, if True then will bin the events prior to merging
        mode - str, the window mode to use, can be 'FF' or 'EFF'
        stack_folder - str, the folder where the stacked files are
        table_suffix - str, the suffix for the individual filenames
    Output:
        table, nobs
        table - Table, the merged events for `ccdnr` and `OBS_ID` during revoultions [`rev_range[0],`rev_range[1]]
        nobs - int, the number of observations included in the stack        
    '''
    if (not os.path.isdir(stacks_folder)):
        print (f'Cannot find folder {stacks_folder} with stacked files. Cannot continue!')
        return None
    if (not (mode in ['FF','EFF'])):
        print (f'Mode {mode} not supported. Cannot continue!')
        return None
    #
    # find all files
    #
    file_pattern = f'{stacks_folder}/{mode}_stacked_*_{table_suffix}.fits.gz'
    stacks = glob.glob(file_pattern)
    if (len(stacks) < 1):
        print (f'Cannot find stacked events for mode {mode} with pattern {file_pattern}. Cannot continue!')
        return None
    #
    if (mode == 'FF'):
        rstep = 100
        revs = np.arange(0,4100,rstep)
    else:
        rstep = 300
        revs = np.arange(0,4100,rstep)
    #
    selected = []
    started = False
    for r0 in revs:
        r1 = r0 + rstep - 1
        if (rev_range[0] <= r1 and rev_range[0] >= r0):
            files = f'{stacks_folder}/{mode}_stacked_{r0}_{r1}_{table_suffix}.fits.gz'
            if (os.path.isfile(files)):
                started = True
                selected.append(files)
            continue
        if (started):
            if (rev_range[1] >= r0 or rev_range[1] >= r1):
                files = f'{stacks_folder}/{mode}_stacked_{r0}_{r1}_{table_suffix}.fits.gz'
                if (os.path.isfile(files)):
                    selected.append(files)    
    if (verbose):
        print (f'Will merge {len(selected)} stacked files for CCDNR: {ccdnr} and mode {mode}')
    #
    tout = None
    for i,j in enumerate(selected):
        tt,nev = extract_events(j,ccdnr=ccdnr,verbose=verbose)
        if (i == 0):
            tout = tt
        else:
            tout = vstack([tout,tt])
    #
    # now filter the revolutions
    #
    rev_mask = (tout['REV'] <= rev_range[1]) & (tout['REV'] >= rev_range[0])
    tout = tout[rev_mask]
    nobs = len(np.unique(tout["OBSID"]))
    #
    if (verbose):
        print (f'Selected {len(tout)} events with rev in [{rev_range[0]},{rev_range[1]}], from {nobs} OBS_IDs')
    # binning if requested
    if (bin_spec):
        tout = bin_events(tout)
        if (verbose):
            print ('Returning the binned spectrum')
    #
    return [tout,nobs]

def fit_cu(events: Table,erange: list=(7000.0,10000.0),ebin: float=5.0,plot_it: bool=False,verbose: bool=False) -> dict:
    '''
    Purpose:
        Fitting a Cu Ka (8038 eV) model, no other lines

    Inputs:
        events - an astropy Table from the stacking results, must have column PI
        erange - a list with low and high energy limints in eV for fit and for display
        ebin - float, the energy binning in eV, will bin the events from the table
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    '''
    emin = erange[0]
    emax = erange[1]
    delta_e = emax - emin
    # for fitting, will use 5 eV binning
    nbins_fit = int(delta_e/ebin)
    hist1, bin_edges1 = np.histogram(events['PI'],bins=nbins_fit,range=erange,density=False)
    fmax = hist1.max()
    # fitting model, linear + Gaussian (Ni Ka) + Gaussian (Cu Ka) + Gaussian (Zn Ka) + Gaussian (Cu Kb)
    #l = models.Linear1D()
    cont = models.Polynomial1D(2)
    g1 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    # get the middle of each bin
    xmid = (bin_edges1[0:-1] + bin_edges1[1:])/2.0
    # exclude the last bin due to some numerical outliers
    xmid = xmid[0:-1]
    hist1 = hist1[0:-1]
    #
    xmodel = cont + g1
    #
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit._calc_uncertainties = True
    maxit = 100
    check = True
    while (check):
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
    chisqr_0 = (residual**2)/fitted(xmid)
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    lines = {'chisqr_r': chisqr_r, 'cuka': [fitted.mean_1.value,fitted.mean_1.std]}
    if (verbose):
        print (f'Chisqr: {chisqr:.1f}, dof: {len(residual)}')
        print (f'Chisqr_reduced: {chisqr_r:.3f}')
        print (lines)
    #
    # return results
    #
    #
    if (plot_it):
        fig, ax = plt.subplots(2,1,figsize=(10,8))
        #
        # per components
        #
        lx = fitted[0](erun)
        g1x = fitted[1](erun)
        #
        #ax[0].hist(events['PI'],bins=nbins_fit,range=erange,histtype='step',label='Data',density=False)
        ax[0].plot(xmid,hist1,label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].grid()
        ax[0].legend()
        #
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        ax[1].plot(xmid,residual,label='Data')
        ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        plt.show()
    return lines

def fit4lines(events: Table,erange: list=(7000.0,10000.0),ebin: float=5.0,plot_it: bool=False,verbose: bool=False) -> dict:
    '''
    Purpose:
        Fitting a 4 lines model for instrumental lines around 8 keV
        Ni Ka (7470 eV), Cu Ka (8038 eV), Zn Ka (8630 eV) and Cu Kb (8900 eV)

    Inputs:
        events - an astropy Table from the stacking results, must have column PI
        erange - a list with low and high energy limints in eV for fit and for display
        ebin - float, the energy binning in eV, will bin the events from the table
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    '''
    emin = erange[0]
    emax = erange[1]
    delta_e = emax - emin
    # for fitting, will use 5 eV binning
    nbins_fit = int(delta_e/ebin)
    hist1, bin_edges1 = np.histogram(events['PI'],bins=nbins_fit,range=erange,density=False)
    fmax = hist1.max()
    # fitting model, linear + Gaussian (Ni Ka) + Gaussian (Cu Ka) + Gaussian (Zn Ka) + Gaussian (Cu Kb)
    #l = models.Linear1D()
    cont = models.Polynomial1D(2)
    g1 = models.Gaussian1D(amplitude=fmax/2.0, mean=7470, stddev=100.0)
    g2 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    g3 = models.Gaussian1D(amplitude=fmax/3.0, mean=8630.0, stddev=100.0)
    g4 = models.Gaussian1D(amplitude=fmax/4.0, mean=8900.0, stddev=100.0)
    # get the middle of each bin
    xmid = (bin_edges1[0:-1] + bin_edges1[1:])/2.0
    # exclude the last bin due to some numerical outliers
    xmid = xmid[0:-1]
    hist1 = hist1[0:-1]
    #
    xmodel = cont + g1 + g2 + g3 + g4
    #
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit._calc_uncertainties = True
    maxit = 100
    check = True
    while (check):
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
    lines = {'chisqr_r': chisqr_r, 'nika': [fitted.mean_1.value,fitted.mean_1.std],'cuka': [fitted.mean_2.value,fitted.mean_2.std],
        'znka': [fitted.mean_3.value,fitted.mean_3.std], 'cukb': [fitted.mean_4.value,fitted.mean_4.std]}
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
        g1x = fitted[1](erun)
        g2x = fitted[2](erun)
        g3x = fitted[3](erun)
        g4x = fitted[4](erun)
        #
        #ax[0].hist(events['PI'],bins=nbins_fit,range=erange,histtype='step',label='Data',density=False)
        ax[0].plot(xmid,hist1,label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].plot(erun,g2x,lw=1,ls='dashed')
        ax[0].plot(erun,g3x,lw=1,ls='dashed')
        ax[0].plot(erun,g4x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('4-lines model')
        ax[0].grid()
        ax[0].legend()
        #
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        ax[1].plot(xmid,ratio,label='Data/Model')
        ax[1].axhline(1.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        plt.show()
    return lines


def fit6lines(events: Table,erange: list=(5600.0,10000.0),ebin: float=5.0,use_column: str = 'PI', 
        plot_it: bool=False, plot_file: str=None, verbose: bool=False,labels: bool=True):
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

def binfit4lines(binned: Table,plot_it: bool=False,verbose: bool=False) -> dict:
    '''
    Purpose:
        Fitting a 4 lines model for fluorescent lines at ~6 to ~9 keV
        Ni Ka (7470 eV), Cu Ka (8038 eV), Zn Ka (8630 eV) and Cu Kb (8900 eV)
        Will not use Mn Ka and Mn Kb

    Inputs:
        binned - an astropy Table from the binned results, must have columns 'bin' and 'count'
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    '''
    #
    fmax = np.max(binned['counts'])
    cont = models.Polynomial1D(2)
    g1 = models.Gaussian1D(amplitude=fmax/2.0, mean=7470, stddev=100.0)
    g2 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    g3 = models.Gaussian1D(amplitude=fmax/3.0, mean=8630.0, stddev=100.0)
    g4 = models.Gaussian1D(amplitude=fmax/4.0, mean=8900.0, stddev=100.0)
    #
    # total model
    #
    xmodel = cont + g1 + g2 + g3 + g4
    #
    # filter the input binned table to contain only bins in [7000,10000] eV
    #
    twork = binned.copy()
    twork = twork[(twork['bin'] >= 7000) & (twork['bin'] <= 10000.0)]
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit._calc_uncertainties = True
    maxit = 100
    check = True
    while (check and maxit <= 2000):
        fitted = fit(xmodel, twork['bin'], twork['counts'],maxiter=maxit,weights=1.0/np.sqrt(twork['counts']))
        if (fitted.stds is None):
            maxit += 100
        else:
            check = False
    #
    if (verbose):
        print (f'Solved in {maxit} iterations')
        print (fitted.stds)
    #
    erun = twork['bin']
    yfit2 = fitted(erun)
    residual = yfit2 - twork['counts']
    ratio = twork['counts']/yfit2
    chisqr_0 = (residual**2)/yfit2
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    lines = {'chisqr_r': chisqr_r, 
             'nika': [fitted.mean_1.value,fitted.mean_1.std],
             'cuka': [fitted.mean_2.value,fitted.mean_2.std],
             'znka': [fitted.mean_3.value,fitted.mean_3.std], 
             'cukb': [fitted.mean_4.value,fitted.mean_4.std]}
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
        g1x = fitted[1](erun)
        g2x = fitted[2](erun)
        g3x = fitted[3](erun)
        g4x = fitted[4](erun)
        #
        #ax[0].hist(events['PI'],bins=nbins_fit,range=erange,histtype='step',label='Data',density=False)
        ax[0].plot(twork['bin'],twork['counts'],label='Data')
        ax[0].plot(erun,yfit2,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].plot(erun,g2x,lw=1,ls='dashed')
        ax[0].plot(erun,g3x,lw=1,ls='dashed')
        ax[0].plot(erun,g4x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('4-lines model')
        ax[0].grid()
        ax[0].legend()
        #
        ax[1].plot(erun,ratio,label='Data/Model')
        ax[1].axhline(1.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].set_ylabel('Data/Model')
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        #ax[1].plot(xmid,residual,label='Data')
        #ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        #ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        plt.show()
    return lines

def binfit5lines(binned: Table,plot_it: bool=False,verbose: bool=False) -> dict:
    '''
    Purpose:
        Fitting a 5 lines model for fluorescent lines at ~6 to ~9 keV
        Mn Ka (5898 eV), Ni Ka (7470 eV), Cu Ka (8038 eV), Zn Ka (8630 eV) and Cu Kb (8900 eV)
        Will not use Mn Kb (6490 eV) as it becomes too faint quckly.

    Inputs:
        binned - an astropy Table from the binned results, must have columns 'bin' and 'count'
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    '''
    #
    fmax = np.max(binned['counts'])
    cont = models.Polynomial1D(2)
    g0 = models.Gaussian1D(amplitude=fmax/2.0, mean=5900, stddev=100.0)
    g1 = models.Gaussian1D(amplitude=fmax/2.0, mean=7470, stddev=100.0)
    g2 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    g3 = models.Gaussian1D(amplitude=fmax/3.0, mean=8630.0, stddev=100.0)
    g4 = models.Gaussian1D(amplitude=fmax/4.0, mean=8900.0, stddev=100.0)
    #
    # total model
    #
    xmodel = cont + g0 + g1 + g2 + g3 + g4
    #
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit._calc_uncertainties = True
    maxit = 100
    check = True
    while (check and maxit <= 2000):
        fitted = fit(xmodel, binned['bin'], binned['counts'],maxiter=maxit,weights=1.0/np.sqrt(binned['counts']))
        if (fitted.stds is None):
            maxit += 100
        else:
            check = False
    #
    if (verbose):
        print (f'Solved in {maxit} iterations')
        print (fitted.stds)
    #
    erun = np.arange(5500,10000.0,5.0)
    yfit1 = fitted(erun)
    yfit2 = fitted(binned['bin'])
    residual = yfit2 - binned['counts']
    ratio = binned['counts']/yfit2
    chisqr_0 = (residual**2)/yfit2
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    lines = {'chisqr_r': chisqr_r, 
             'mnka': [fitted.mean_1.value,fitted.mean_1.std], 
             'nika': [fitted.mean_2.value,fitted.mean_2.std],
             'cuka': [fitted.mean_3.value,fitted.mean_3.std],
             'znka': [fitted.mean_4.value,fitted.mean_4.std], 
             'cukb': [fitted.mean_5.value,fitted.mean_5.std]}
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
        #
        #ax[0].hist(events['PI'],bins=nbins_fit,range=erange,histtype='step',label='Data',density=False)
        ax[0].plot(binned['bin'],binned['counts'],label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g0x,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].plot(erun,g2x,lw=1,ls='dashed')
        ax[0].plot(erun,g3x,lw=1,ls='dashed')
        ax[0].plot(erun,g4x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('5-lines model')
        ax[0].grid()
        ax[0].legend()
        #
        ax[1].plot(binned['bin'],ratio,label='Data/Model')
        ax[1].axhline(1.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].set_ylabel('Data/Model')
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        #ax[1].plot(xmid,residual,label='Data')
        #ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        #ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        plt.show()
    return lines

def binfit6lines(binned: Table,plot_it: bool=False,verbose: bool=False) -> dict:
    '''
    Purpose:
        Fitting a 6 lines model for instrumental lines around 8 keV
        Mn Ka (5898 eV) Mn Kb (6490 eV), Ni Ka (7470 eV), Cu Ka (8038 eV), Zn Ka (8630 eV) and Cu Kb (8900 eV)

    Inputs:
        binned - an astropy Table from the binned results, must have columns 'bin' and 'count'
        plot_it - boolean, if plot is to be displayed
        verbose - boolean, if to be verbose
    
    Outputs: 
        a dict with key the line name and best-fit energy and its uncertainty

    '''
    #
    fmax = np.max(binned['counts'])
    cont = models.Polynomial1D(2)
    g0 = models.Gaussian1D(amplitude=fmax/2.0, mean=5900, stddev=100.0)
    g1 = models.Gaussian1D(amplitude=fmax/2.0, mean=6490, stddev=100.0)
    g2 = models.Gaussian1D(amplitude=fmax/2.0, mean=7470, stddev=100.0)
    g3 = models.Gaussian1D(amplitude=fmax, mean=8000.0, stddev=100.0)
    g4 = models.Gaussian1D(amplitude=fmax/3.0, mean=8630.0, stddev=100.0)
    g5 = models.Gaussian1D(amplitude=fmax/4.0, mean=8900.0, stddev=100.0)
    #
    # total model
    #
    xmodel = cont + g0 + g1 + g2 + g3 + g4 + g5
    #
    # will fit, excluding first and last bin to avoid numerical effects
    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    #fit._calc_uncertainties = True
    maxit = 100
    check = True
    while (check and maxit <= 2000):
        fitted = fit(xmodel, binned['bin'], binned['counts'],maxiter=maxit,weights=1.0/np.sqrt(binned['counts']))
        if (fitted.stds is None):
            maxit += 100
        else:
            check = False
    #
    if (verbose):
        print (f'Solved in {maxit} iterations')
        print (fitted.stds)
    #
    erun = np.arange(5500,10000.0,5.0)
    yfit1 = fitted(erun)
    yfit2 = fitted(binned['bin'])
    residual = yfit2 - binned['counts']
    ratio = binned['counts']/yfit2
    chisqr_0 = (residual**2)/yfit2
    chisqr = np.sum(chisqr_0)
    chisqr_r = chisqr/len(residual)
    lines = {'chisqr_r': chisqr_r, 
             'mnka': [fitted.mean_1.value,fitted.mean_1.std], 
             'mnkb': [fitted.mean_2.value,fitted.mean_2.std], 
             'nika': [fitted.mean_3.value,fitted.mean_3.std],
             'cuka': [fitted.mean_4.value,fitted.mean_4.std],
             'znka': [fitted.mean_5.value,fitted.mean_5.std], 
             'cukb': [fitted.mean_6.value,fitted.mean_6.std]}
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
        ax[0].plot(binned['bin'],binned['counts'],label='Data')
        ax[0].plot(erun,yfit1,label=fr'Best fit ($\chi^2_r = $ {chisqr_r:.2f})')
        ax[0].plot(erun,lx,lw=1,ls='dashed')
        ax[0].plot(erun,g0x,lw=1,ls='dashed')
        ax[0].plot(erun,g1x,lw=1,ls='dashed')
        ax[0].plot(erun,g2x,lw=1,ls='dashed')
        ax[0].plot(erun,g3x,lw=1,ls='dashed')
        ax[0].plot(erun,g4x,lw=1,ls='dashed')
        ax[0].plot(erun,g5x,lw=1,ls='dashed')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('6-lines model')
        ax[0].grid()
        ax[0].legend()
        #
        ax[1].plot(binned['bin'],ratio,label='Data/Model')
        ax[1].axhline(1.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].set_ylabel('Data/Model')
        #ax[1].plot(xmid,chisqr_0,label=r'$\chi^2_r$')
        #ax[1].plot(xmid,residual,label='Data')
        #ax[1].axhline(0.0,linewidth=2,color='black',linestyle='dashed')
        ax[1].grid()
        #ax[1].set_ylabel('Residual (counts)')
        ax[1].set_xlabel('PI (eV)')
        plt.show()
    return lines


# ## Test the code

def test_unbinned():
    wdir = '/xdata/xcaldata/XMM/IVAN/sanders/stacks_0055_0056'
    rrange = (3000,3299)
    myccd = 3
    tx = merge_stacks(myccd,rrange,mode='FF',bin_spec=False,stacks_folder=wdir)
    results4 = fit4lines(tx[0],plot_it=True)
    print (results4)
    results6 = fit6lines(tx[0],plot_it=True,erange=(5600,10000.0))
    print (results6)
    #

def test_binned():
    wdir = '/xdata/xcaldata/XMM/IVAN/sanders/stacks_0055_0056'
    rrange = (3000,3299)
    myccd = 3
    tx = merge_stacks(myccd,rrange,mode='FF',bin_spec=True,stacks_folder=wdir)
    results4 = binfit4lines(tx[0],plot_it=True)
    print (results4)
    results6 = binfit6lines(tx[0],plot_it=True)
    print (results6)
    #