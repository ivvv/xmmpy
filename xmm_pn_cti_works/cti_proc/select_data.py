
import numpy as np

# global
SIG2FWHM = 2.0*np.sqrt(2.0*np.log(2.0))


#%%

def select_data(dataframe,ccd,rawy_range=(1,200),filter_select=None):
    """Make a selection of data from the input dataframe and return the selected dataframe
    Parameters
    ----------
        dataframe : dataframe, mandatory 
            the pandas dataframe with the Cu Kalpha fit results (from Michael Smith monitoring run). Produced by `ff_monitoring_work2.ipynb`
        ccd : int, mandatory 
            the EPIC-pn CCD number (from 1 to 12)
        rawy_range: list, optional
            the RAWY range selection, default the full CCD range (1,200)
        filter_select : str
            if not None, then a selection on filter wheel is requested, can be one of 'CalClosed', 'CalMedium', 'CalThick', 'CalThin1', 'Closed',
           'Medium', 'Thick', 'Thin1', 'Thin2'. If None, then all are selected.
    Output
    ------
        df_out, pandas dataframe
            A new dataframe with selected records, sorted on `delta_time`

    Method
    ------
        First, a selection based on CCD and RAWY range is done.
        Then the further filtering based on 
            * best-fit Gaussian line sigma, within (16,84)% quantiles
            * exposure time (>= 10 ks)
            * number of dicarded lines (<= 300), only applied for FF mode.
            * best-fit line energy mean error ( <= 20 eV) and neither the upper or lower error bar is zero.
            * if filter_select is used, then also select on filter.
        The filtering is just to discard bad fit or poor fit results.
        And we discard duplicates (if any) based on the `delta_time` (time in years since 2000-01-01) and finally
        sort on `delta_time`.
         
    """
    df_ccd = dataframe[(dataframe.ccd == ccd) & (dataframe.rawy0 == rawy_range[0]) & (dataframe.rawy1 == rawy_range[1])]
    ntot, _ = df_ccd.shape
    df_ccd.xmode = dataframe.xmode
    #
    # get the quantile distribution (16,50,84) of best fit Gaussian line sigma
    # will use th elower and upper quantile to filter the bad fit results
    #
    qq = np.quantile(df_ccd.sigma,(0.16,0.84))
    fwhm = np.rint(qq*SIG2FWHM).astype(int)
    qq = np.rint(qq).astype(int)
    #
    if (df_ccd.xmode == 0):
        xmode = 'FF'
        if (filter_select is not None):
            df_out = df_ccd[(df_ccd.ccd == ccd)  & (df_ccd.expo_time >= 10000.0) & 
                             (df_ccd.rawy0 == rawy_range[0]) & (df_ccd.rawy1 == rawy_range[1]) &
                             ((df_ccd.sigma >= qq[0]) & (df_ccd.sigma <= qq[1])) & 
                             #(df_ccd.chi2r <= 3.0) & (df_ccd.dof >= 10) & 
                             ((df_ccd.energy_err1 + df_ccd.energy_err2)/2.0 <= 20.0) & 
                             (df_ccd.ndl <= 300.0) & (df_ccd['filter'] == filter_select) & 
                             (df_ccd.energy_err1*df_ccd.energy_err2 > 0.0)].drop_duplicates('delta_time')
        else:
            df_out = df_ccd[(df_ccd.ccd == ccd) & (df_ccd.expo_time >= 10000.0) & 
                             (df_ccd.rawy0 == rawy_range[0]) & (df_ccd.rawy1 == rawy_range[1]) &
                             ((df_ccd.sigma >= qq[0]) & (df_ccd.sigma <= qq[1])) & 
                             #(df_ccd.chi2r <= 3.0) & (df_ccd.dof >= 10) & 
                             ((df_ccd.energy_err1 + df_ccd.energy_err2)/2.0 <= 20.0) & 
                             (df_ccd.ndl <= 300.0) &
                             (df_ccd.energy_err1*df_ccd.energy_err2 > 0.0)].drop_duplicates('delta_time')
    elif (df_ccd.xmode == 1):
        xmode = 'EFF'
        if (filter_select is not None):
            df_out = df_ccd[(df_ccd.ccd == ccd) & (df_ccd.expo_time >= 10000.0) & 
                             (df_ccd.rawy0 == rawy_range[0]) & (df_ccd.rawy1 == rawy_range[1]) &
                             ((df_ccd.sigma >= qq[0]) & (df_ccd.sigma <= qq[1])) & 
                             #(df_ccd.chi2r <= 3.0) & (df_ccd.dof >= 10) & 
                             ((df_ccd.energy_err1 + df_ccd.energy_err2)/2.0 <= 20.0) & 
                             (df_ccd['filter'] == filter_select) & 
                             (df_ccd.energy_err1*df_ccd.energy_err2 > 0.0)].drop_duplicates('delta_time')
        else:
            df_out = df_ccd[(df_ccd.ccd == ccd) & (df_ccd.expo_time >= 10000.0) & 
                             (df_ccd.rawy0 == rawy_range[0]) & (df_ccd.rawy1 == rawy_range[1]) &
                             ((df_ccd.sigma >= qq[0]) & (df_ccd.sigma <= qq[1])) & 
                             #(df_ccd.chi2r <= 3.0) & (df_ccd.dof >= 10) & 
                             ((df_ccd.energy_err1 + df_ccd.energy_err2)/2.0 <= 20.0) & 
                             (df_ccd.energy_err1*df_ccd.energy_err2 > 0.0)].drop_duplicates('delta_time')
        #
    else:
        print (f'Cannot process mode={df_ccd.xmode}, only mode=0 (FF) or mode=1 (EFF).')
        return None
    #
    _ = df_out.sort_values(by='delta_time',inplace=True)
    df_out.xmode = dataframe.xmode
    #
    return df_out
