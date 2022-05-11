
from dataclasses import dataclass
import numpy as np

from astropy import stats
from astropy.table import QTable

from scipy.stats import median_abs_deviation as mad

import matplotlib.pylab as plt

from select_data import select_data

# global
line0 = 8040.0 # Elab in eV

#%%
def check_ltc_residuals(dataframe,ccd,rawy_range=(1,200),binned=None,filter_select=None, plot_it=True, png_file=None,title=''):
    """Validation for the LTC correction by means of checking the residuals
    
     Parameters
     ----------
        dataframe : dataframe, mandatory 
            the pandas dataframe with the Cu Kalpha fit results (from Michael Smith monitoring run). Produced by `ff_monitoring_work2.ipynb`
        ccd : int, mandatory 
            the EPIC-pn CCD number (from 1 to 12)
        rawy_range : list
            the RAWY range to select, can be (1,200) and then can be (x,x+19) for x in range(1,201,20)
        binned: float, optional
            bin the data grouping with binned years, if None, then no binning
        filter_select : str
            if not None, then a selection on filter wheel is requested, can be one of 'CalClosed', 'CalMedium', 'CalThick', 'CalThin1', 'Closed',
           'Medium', 'Thick', 'Thin1', 'Thin2'. If None, then all are selected.
        plot_it : bool
            if set, will plot the results.
        png_file : str
            is set, then the plotting results will be saved to this file.
        title : str
            Text to append to the end of the plot title, e.g. the version or other comment to apper on the plot title
     Output
     ------
        output: dict
            {'ccd': []}

    Method
    ------
            
    Modification history
    --------------------
    
        Created 17 Mar 2021, Ivan Valchanov, XMM SOC

    """
    #
    ntot = np.count_nonzero((dataframe.ccd == ccd) & (dataframe.rawy0 == rawy_range[0]) & (dataframe.rawy1 == rawy_range[1]))
    #
    xtab = select_data(dataframe,ccd,rawy_range=rawy_range,filter_select=filter_select)
    ntab = len(xtab)
    xmode = xtab.xmode
    if (filter_select is not None): 
        print (f"CCD {ccd}, {xmode} mode, filter {filter_select}: filtered {ntab} results out of {ntot}")
    else:
        print (f"CCD {ccd}, {xmode} mode: filtered {ntab} results out of {ntot}")
    #
    #line = dataframe.line
    #
    xin = xtab.delta_time
    residual = (xtab.energy - line0) # in eV
    residual_err = (xtab.energy_err1 + xtab.energy_err2)/2.0 # in eV
    qmean = np.mean(residual)
    qstd = np.std(residual)
    xstat = stats.sigma_clipped_stats(residual, sigma=3, maxiters=3)
    #
    if (binned is not None):
        # add those as columns in the dataframe
        qt = QTable.from_pandas(xtab)
        qt['residual'] = residual
        qt['residual_err'] = residual_err
        #
        year_bin = np.trunc(qt['delta_time']/binned)
        year_run = np.unique(year_bin)
        #year_bin = np.trunc(qt['delta_time']/binned)
        dat_grouped = qt.group_by(year_bin)
        #
        dat_binned = dat_grouped.groups.aggregate(np.median)
        dat_binned_std = dat_grouped.groups.aggregate(mad)
        xin_bin = dat_binned['delta_time']
        yin_bin = dat_binned['residual']
        yin_bin_err = dat_binned_std['residual']
    #
    # prepare the output
    #
    output = [ccd,xmode,rawy_range[0],rawy_range[1],xstat[0],xstat[2],ntab,year_run.data,xin_bin.data,yin_bin.data,yin_bin_err.data]
    #
    # plotting
    #
    if (plot_it):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.errorbar(xin,residual,yerr=residual_err,fmt='o',label=f'CCDNR {ccd}',zorder=0)
        if (binned is not None):
            ax.step(year_run,yin_bin,where='pre',zorder=2,color='cyan',label='Per bin median')
            #ax.step(xin_bin,yin_bin,where='mid',zorder=2,color='cyan',label='Per bin median')
            ax.errorbar(xin_bin,yin_bin,yerr=yin_bin_err,fmt='o',color='cyan',zorder=1)
        ax.axhline(0.0,linestyle='dashed',linewidth=3,color='red',zorder=1)
        ax.axhline(20.0,linestyle='dotted',linewidth=2,color='red',zorder=1)
        ax.axhline(-20.0,linestyle='dotted',linewidth=2,color='red',zorder=1)
        ax.text(0.1,0.9,fr'mean={qmean:.1f} eV, st.dev.={qstd:.1f} eV',fontsize=14,transform=ax.transAxes)
        ax.text(0.1,0.8,fr'mean={xstat[0]:.1f} eV, st.dev.={xstat[2]:.1f} eV (3-$\sigma$ clipped)',fontsize=14,transform=ax.transAxes)
        ax.set_xlim((0.0,22.0))
        ax.set_ylim((-100.0,100.0))
        ax.grid(True)
        ax.legend(loc=3)
        ax.set_title(f"Cu-Ka data for EPIC-PN CCD{ccd:02}, mode={xmode}, RAWY in [{rawy_range[0]},{rawy_range[1]}], {title}")
        ax.set_ylabel(r"E$_{corr}$ - E$_{lab}$ (eV)")
        ax.set_xlabel("Time since 2000-01-01 (years)")
        if (png_file is not None):
            plt.savefig(png_file,dpi=100)
            plt.show()
            plt.close()
    return output
