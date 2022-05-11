import numpy as np

from select_data import select_data

# global
line0 = 8040.0 # Elab in eV

#%%
def ltc_residuals_outliers(dataframe,ccd,threshold = 50.0, rawy_range=(1,200),filter_select=None):
    """Output of outliers for the LTC correction residuals
    
     Parameters
     ----------
        dataframe : dataframe, mandatory 
            the pandas dataframe with the Cu Kalpha fit results (from Michael Smith monitoring run). Produced by `ff_monitoring_work2.ipynb`
        ccd : int, mandatory 
            the EPIC-pn CCD number (from 1 to 12)
        threshold : float
            the threshold in +/- eV above and below which to consider the residual as outlier.
        rawy_range : list
            the RAWY range to select, can be (1,200) and then can be (x,x+19) for x in range(1,201,20)
        filter_select : str
            if not None, then a selection on filter wheel is requested, can be one of 'CalClosed', 'CalMedium', 'CalThick', 'CalThin1', 'Closed',
           'Medium', 'Thick', 'Thin1', 'Thin2'. If None, then all are selected.
        
     Output
     ------
        table with the outliers

    Method
    ------
            
    Modification history
    --------------------
    
        Created 09 Mar 2021, Ivan Valchanov, XMM SOC

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
    #xin = xtab.delta_time
    #
    residual = xtab.energy - line0
    xtab["residual"] = residual
    #xstat = stats.sigma_clip(residual, sigma=3, maxiters=3)
    outliers = (np.abs(residual) >= threshold)
    #
    out = xtab[outliers]
    print (f'Found {len(out)} outliers with residuals above ±{threshold} eV')
    return out
#
