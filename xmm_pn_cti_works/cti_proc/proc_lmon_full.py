# 
# In this script, for the selected mode (FF or EFF), will concatenate all results for a particular line and for a given RAWY range into one single file.
# 
# Will use the previously merged summary files and add the fit results to those.

# ## The contents of a `lmon` file is the following:
# 
# ```
# xbinrange_lo
# xbinrange_hi
# ybinrange_lo
# ybinrange_hi
# gauss centroid
# gauss centroid confidence_lo
# gauss centroid confidence_hi
# gauss width
# gauss width confidence_lo
# gauss width confidence_hi
# gauss area
# gauss area confidence_lo
# gauss area confidence_hi
# powerlaw slope
# powerlaw slope confidence_lo
# powerlaw slope confidence_hi
# powerlaw norm
# powerlaw norm confidence_lo
# powerlaw norm confidence_hi
# fit status flag 0
# fit status flag 1
# fit status flag 2
# fit status flag 3
# minimum fit statistic
# dof
# ```
# 
# ## History:
# * Created: 08 Oct 2021, Ivan Valtchanov, based on old notebooks, script is better for batch processing.

import os
import glob
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

from astropy.table import QTable
from astropy import units as u

from datetime import datetime
time0 = datetime.strptime("2000-01-01T00:00:00","%Y-%m-%dT%H:%M:%S")

home = os.path.expanduser('~')
#%%
#
# 
def proc_lmon_full(merged_summaries_file: str, inst_mode: str = 'FF', xline: str = 'Cu-Ka', 
    data_dir: str = "./", output_file: str = "output.csv",verbose: bool =True) -> pd.DataFrame:
    """
        Merge all lmon fit results for a given line and mode, for each CCD and the full RAWY range (1,200)

        Input Parameters:
            df : DataFrame
                Pandas DataFrame with the merged summaries, will be used to select the OBS_IDs for the `xmode`
            inst_mode : str
                The instrument mode, can be 'FF' or 'EFF'
            xline : str
                The label of the line, as in the output files from Michael, can be 'Cu-Ka', 'Mn-Ka' or 'Al-Ka'
            data_dir : str
                The absolute path to the monitoring results folder, for example "/xdata/xcaldata/XMM/PN/CTI/dat_0062_sci"
            output_file : str
                The absolute path to the output file where the results will be saved
            verbose : bool
                If True will print some verbose info
        Output:
            the merged results in a pandas DataFrame
    """
    #
    # lines of interest and their rest-frame energies in eV
    #
    lines0 = {'Cu-Ka': 8038.0, 'Mn-Ka': 5899.0, 'Al-Ka': 1486.0}
    #
    if (not os.path.isfile(merged_summaries_file)):
        print (f'File with merged summaries {merged_summaries_file} not found. Cannot continue.')
        return None
    #
    if (not os.path.isdir(data_dir)):
        print (f'Data dir {data_dir} not found. Cannot continue.')
        return None
    #
    if (xline not in lines0.keys()):
        print (f"Line {xline} not in list with rest-frame energy, please add it and run again.")
        return None
    #
    if (inst_mode == 'EFF'):
        select_mode = "PrimeFullWindowExtended"
    elif (inst_mode == 'FF'):
        select_mode = "PrimeFullWindow"
    else:
        print ('Only inst_mode=\'FF\' or \'EFF\' supported')
        return None
    df = QTable.read(merged_summaries_file)
    #
    df.sort('rev')
    #
    df = df[df['mode'] == select_mode]
    #
    nt = len(df)
    print (f"Will process {len(df)} {inst_mode} mode observations")
    print ("Doing line",xline)
    #
    all_cols = ['obsid','expo_name','rev','delta_time','omode','filter','expo_time','ccd', 'mipsel','maxmip','med_ndl','ndl','ndl_err']
    # from meanXY.txt file
    all_cols.extend(["rawx0","rawx1","rawy0","rawy1","rawx_mean","rawx_med","rawx_16","rawx_84",
                       "rawy_mean","rawy_med","rawy_16","rawy_84","nevents"])
    # feom lmon file
    all_cols.extend(["energy","energy_lo","energy_hi",
               "sigma","sigma_lo","sigma_hi","area","area_lo","area_hi",
               "pw_order","pw_slope","pw_slope_lo","pw_slope_hi","pw_norm","pw_norm_lo","pw_norm_hi",
               "fit_flag0","fit_flag1","fit_flag2","fit_flag3","fit_stat","dof"])
    #    
    bore_ccds = [1,4,7,10]
    # set up the CCD numbers and the corresponding quadrants
    quad = {1: '0', 2: '0', 3: '0', 
            4: '1', 5: '1', 6: '1', 
            7: '2', 8: '2', 9: '2', 
            10: '3', 11: '3', 12: '3',}
    #
    t = QTable(names=all_cols, dtype=[int, str, int, float, str, str, float, int, int, int, 
        float, float, float, int, int, int, int, float, float, float, 
        float, float, float, float, float, int, float, float, float, 
        float, float, float, float, float, float, float, float, float, 
        float, float, float, float, int, int, int, int, float, int])
    #print (t.colnames)
    #
    start_time = time.time()
    #
    for i in tqdm(range(nt),desc='Processing obs'):
        iobs = df['obsid'][i]
        irev = df['rev'][i]
        inexp = df['expid'][i]
        istart = df['tstart'][i]
        iexpo = df['texpo'][i]
        imode = df['mode'][i]
        ifilter = df['filt'][i]
        #
        stime = datetime.strptime(istart,"%Y-%m-%dT%H:%M:%S")
        delta_time = (stime-time0).total_seconds()/(365.0*24.0*3600.0) # in years
        #
        part1 = [iobs,inexp,irev,delta_time,imode,ifilter,iexpo]
        #
        part2 = []
        for iccd in np.arange(1,13):
            #
            part2 = part1.copy()
            resfile = f"{data_dir}/{iobs:010}/{iobs:010}{inexp}*lmonCCD{iccd:02}_{xline}.txt"
            xfile = glob.glob(resfile)
            # need to do this as there is inconsistency in the file names
            axline = xline.split('-')[0]
            # _meanXY.txt
            rawxy_file1 = f"{data_dir}/{iobs:010}/{iobs:010}{inexp}*lmonCCD{iccd:02}_{axline}_meanXY.txt"
            file_means = glob.glob(rawxy_file1)
            #
            # check if this CCD has results
            #
            if ((len(xfile) < 1) or (len(file_means) < 1)):
                print (f'No data for RAWY in (1,200) for CCD {iccd}, {iobs:010}')
                continue
            # now add quadrant specific parameters
            ndl = df[f'ndisclin_mean{quad[iccd]}'][i]
            med_ndl = df[f'ndisclin_med{quad[iccd]}'][i]
            ndl_err = df[f'ndisclin_std{quad[iccd]}'][i]
            mipsel = df[f'mipsel{quad[iccd]}'][i]
            maxmip = df[f'maxmip{quad[iccd]}'][i]
            #
            part2.extend([iccd,mipsel,maxmip,med_ndl,ndl,ndl_err])
            #
            # read the meanXY file as text
            #
            out1_line = []
            with open(file_means[0],'r') as mm:
                qlines = mm.readlines()
            for qline in qlines:
                qx = qline.split()
                if ((qx[2] == '1') and (qx[3] == '200')):
                    out1_line = qx
                    break
            # skip if no results for RAWY (1,200) are available
            if (len(out1_line) < 1):
                print (f'No meanXY data for RAWY in (1,200) for CCD {iccd}')
                continue
            #
            out2_line = []
            with open(xfile[0],'r') as mm:
                qlines = mm.readlines()
            for qline in qlines:
                qx = qline.split()
                if ((qx[2] == '1') and (qx[3] == '200')):
                    out2_line = qx
                    break
            #
            # skip if no results for RAWY (1,200) are available
            if (len(out2_line) < 1):
                print (f'No lmon data for RAWY in (1,200) for CCD {iccd}')
                continue
            #
            # now extend the array
            #
            out1_line.extend(out2_line[4:])
            part2.extend(out1_line)
            #
            t.add_row(part2)
            #
    tx = _convert_adu(t)
    #
    return t

def _convert_adu(dataframe):
    #
    # convert the columns from ADU to eV and set units to some columns
    #
    dataframe['energy'] *= 5.0  # energy from ADU to eV
    dataframe['energy_err1'] = np.abs(dataframe['energy'] - 5.0*dataframe['energy_lo'])  # energy error in eV
    dataframe['energy_err2'] = np.abs(dataframe['energy'] - 5.0*dataframe['energy_hi'])  # energy error in eV
    #
    dataframe['energy'].unit = u.eV
    dataframe['energy_err1'].unit = u.eV
    dataframe['energy_err2'].unit = u.eV

    #
    dataframe['sigma'] *= 5.0  # energy from ADU to eV
    dataframe['sigma_err1'] = np.abs(dataframe['sigma'] - 5*dataframe['sigma_lo'])  # sigma from ADU to eV
    dataframe['sigma_err2'] = np.abs(dataframe['sigma'] - 5*dataframe['sigma_hi'])  # sigma from ADU to eV
    #
    dataframe['sigma'].unit = u.eV
    dataframe['sigma_err1'].unit = u.eV
    dataframe['sigma_err2'].unit = u.eV
    #
    dataframe['area'] /= dataframe['expo_time']
    dataframe['area_err1'] = np.abs(dataframe['area'] - dataframe['area_lo']/dataframe['expo_time'])
    dataframe['area_err2'] = np.abs(dataframe['area'] - dataframe['area_hi']/dataframe['expo_time'])
    #
    # add other units
    dataframe['delta_time'].unit = u.year
    dataframe['expo_time'].unit = u.s
    #
    return dataframe
