
import numpy as np
from tqdm import tqdm
from astropy.table import Table

from fit_single_line import fit_single_line

line0 = 5898.8

def run_fit_for_mnka(table,bin_size=5.0,eranges=[5500.0,6500.0],use_column='PI',verbose=False):
    #
    # Run the fit on the stacked spectra, the spectra are binned with bin_size, bin_size is in keV, default 5 eV
    # for fixed RAWX, if rawx='all' then will run on all RAWX from 1 to 64.
    #
    # model can be '4line' or 'single'
    #
    if (not isinstance(table,Table)):
        print (f'Input is not an astropy.table.Table, your input is {table.__class__}')
        return None
    if (not (use_column in ['PI','PI_CORR'])):
        print ('Only columns PI or PI_CORR can be used')
        return None
    #
    bins = np.arange(eranges[0],eranges[1],bin_size)
    # make the RAWX and RAWY grids
    rawx = np.arange(1,65,1)
    rawy_array = np.arange(20,200,20)
    # below RAWY=12 no data
    rawy_array[0] = 12
    #
    # create the output arrays, fill with NaNs
    #
    results = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    results_nevts = np.full((len(rawx),len(rawy_array)),0,dtype=int)
    results_err = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    results_redchi = np.full((len(rawx),len(rawy_array)),np.nan,dtype=np.single)
    #
    # Loop on RAWX from 1 to 64
    #
    for irawx in tqdm(rawx,desc='Processing RAWX'):
        #
        ix = irawx - 1
        #
        # Loop on RAWY, from 12 to 200 with step 20 (9 bins)
        # Note, there are no events at RAWY < 12
        #
        for iy,irawy in enumerate(rawy_array):
            rawy_start = irawy
            rawy_end = irawy + 20
            tmask = (table['RAWY'] < rawy_end) & (table['RAWY'] >= rawy_start) & (table['RAWX'] == irawx)
            tsel = table[tmask]
            out = fit_single_line(tsel,line_c=line0,erange=[5500.0,6500.0],ebin=5,plot_it=False,use_column=use_column,verbose=verbose)
            if (out is None):
                continue
            # Saving the result (best fit Mn-Ka)
            results[ix,iy] = out['line_c'][0]
            # error on best-fit energy
            results_err[ix,iy] = out['line_c'][1]
            # reduced ChiSqr
            results_nevts[ix,iy] = out['nevts']
            results_redchi[ix,iy] = out['chisqr_r']
            #
    #
    # now make it a full RAW CCD with residuals in eV
    #
    yrr = np.full((64,200),np.nan,dtype=np.single)
    yrr_nevts = np.full((64,200),0,dtype=int)
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
            yrr[jx,i0:i1] = (results[jx,jy] - line0)
            yrr_nevts[jx,i0:i1] = results_nevts[jx,jy]
            yrr_err[jx,i0:i1] = results_err[jx,jy]
            yrr_redchi[jx,i0:i1] = results_redchi[jx,jy]
    yrr[:,:12] = np.nan
    yrr_err[:,:12] = np.nan
    yrr_nevts[:,:12] = 0
    yrr_redchi[:,:12] = np.nan
    #
    return (yrr,yrr_err,yrr_redchi,yrr_nevts)
