import os
import glob
import tempfile
from datetime import datetime
from tqdm import tqdm

import pandas as pd

# global time reference for XMM
time0 = datetime.strptime("2000-01-01T00:00:00","%Y-%m-%dT%H:%M:%S")

#%%
def merge_summaries(root_dir: str,output_file: str=None) -> pd.DataFrame:
    """Merging all summary files from PN monitoring results in a table
    
    Each sub-folder of `root_dir` contains a summary file with information about the 
    observation. I combine all this info in a table and save it for easy access to 
    the relevant `OBS_ID`.
    
    Also add a new column with the time (in years) since 2000-01-01 used in XMM calibration.
    
    """
    #
    print (f'Collecting the available summary files in {root_dir}, can take time... please wait.')
    sumfiles = glob.glob(f"{root_dir}/**/*smry.txt",recursive=True)
    nsums = len(sumfiles)
    print (f"Found {nsums} summary files in {root_dir}")
    #
    # will concatenate all smry.txt files into one temporary file and then will put it in pandas DataFrame and 
    # save as CSV
    #
    with tempfile.NamedTemporaryFile(mode='w') as fp:
        for sumfile in tqdm(sumfiles,desc='Collecting the summaries'):
            with open(sumfile,'r') as sfile:
                fp.write(sfile.read())
        #
        # now read as pandas dataframe
        #
        colnames = ["rev","obsid","expid","mode","filt","tstart","tend","texpo","mvcratio", # (a rough measure of the ratio of counts in the MnKa versus continuum)
                    "qboxt0","qboxt1","qboxt2","qboxt3", # x 4 (electronics quadrant box temperatures)
                    "ndisclin_mean0","ndisclin_mean1","ndisclin_mean2","ndisclin_mean3", #x 4
                    "mipsel0","mipsel1","mipsel2","mipsel3", #x 4 (parameter for on-board MIP rejection algorithm)
                    "maxmip0","maxmip1","maxmip2","maxmip3", #x 4 (parameter for on-board MIP rejection algorithm)
                    "ndisclin_med0","ndisclin_med1","ndisclin_med2","ndisclin_med3", #median x 4
                    "ndisclin_std0","ndisclin_std1","ndisclin_std2","ndisclin_std3"] #, stddev x 4

        df = pd.read_csv(fp.name,delimiter='\s+',header=None,skip_blank_lines=True,names=colnames)
    #
    # now calculate the time_delta, the difference in years from observation start and 2000-01-01
    #
    stime = [(datetime.strptime(x,"%Y-%m-%dT%H:%M:%S")-time0).total_seconds()/(365.0*24.0*3600.0)  for x in df.tstart]
    df.insert(6,"delta_time",pd.Series(stime,index=df.index))
    #
    print (f'Last observation t={df.delta_time.max():.2f} years')
    if (output_file is not None):
        df.to_csv(output_file)
    fp.close()
    return df

def merge_summaries_old(root_dir,output_file=None):
    """Merging all summary files from PN monitoring results
    
    Each sub-folder of `root_dir` contains a summary file with information about the 
    observation. I combine all this info in a table and save it for easy access to 
    the relevant `OBS_ID`.
    """
    #
    sumfiles = glob.glob(f"{root_dir}/**/*smry.txt",recursive=True)
    nsums = len(sumfiles)
    print (f"Found {nsums} summary files in {root_dir}")
    #
    with tempfile.NamedTemporaryFile(mode='w') as fp:
        for i in range(nsums):
            sumfile = sumfiles[i]
            iobs = os.path.basename(sumfile)[0:10]
            with open(sumfile,'r') as sfile:
                fp.write(sfile.read())
        #
        # now read as pandas dataframe
        #
        colnames = ["rev","obsid","expid","mode","filt","tstart","tend","texpo",\
                    "mvcratio", # (a rough measure of the ratio of counts in the MnKa versus continuum)
                    "qboxt0","qboxt1","qboxt2","qboxt3", # x 4 (electronics quadrant box temperatures)
                    "ndisclin_mean0","ndisclin_mean1","ndisclin_mean2","ndisclin_mean3", #x 4
                    "mipsel0","mipsel1","mipsel2","mipsel3", #x 4 (parameter for on-board MIP rejection algorithm)
                    "maxmip0","maxmip1","maxmip2","maxmip3", #x 4 (parameter for on-board MIP rejection algorithm)
                    "ndisclin_med0","ndisclin_med1","ndisclin_med2","ndisclin_med3", #median x 4
                    "ndisclin_std0","ndisclin_std1","ndisclin_std2","ndisclin_std3"] #, stddev x 4
        #
        df = pd.read_csv(fp.name,delimiter='\s+',header=None,skip_blank_lines=True,names=colnames)
    #
    # now calculate the time_delta, the difference in years from observation start and 2000-01-01
    #
    stime = [(datetime.strptime(x,"%Y-%m-%dT%H:%M:%S")-time0).total_seconds()/(365.0*24.0*3600.0)  for x in df.tstart]
    df.insert(6,"delta_time",pd.Series(stime,index=df.index))
    #
    print (f'Last observation t={df.delta_time.max():.2f} years')
    if (output_file is not None):
        df.to_csv(output_file)
    fp.close()
    return df
