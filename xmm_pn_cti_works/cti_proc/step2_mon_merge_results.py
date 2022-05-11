# ## History:
# * Created: 08 Oct 2021, Ivan Valtchanov, based on old notebooks, script is better for batch processing.
# * modified it on 11 may 2022, adapted to the new package structure

import os
import argparse

from proc_lmon_full import proc_lmon_full

home = os.path.expanduser('~')
#%%
#
# parse the input parameters
#
parser = argparse.ArgumentParser(description='Merge all summary and lmod  files from monitoring processing for long-term CTI')
parser.add_argument('data_folder', type=str, 
                    help='The absolute path to the root folder for the monitoring results, like "/xdata/xcaldata/XMM/PN/CTI/dat_0062_sci"')
parser.add_argument('summaries_file', type=str, 
                    help='The absolute path to the summaries file (from step1)')
parser.add_argument('window_mode', type=str, 
                    help='The observing mode, can be "FF" or "EFF"')
parser.add_argument('xray_line', type=str, 
                    help='The line to process, can be "Cu-Ka", "Mn-Ka" or "Al-Ka"')
parser.add_argument('output_file', type=str,
                    help='The name of the merged output FITS file. it will be overwritten!')
parser.add_argument('-s','--skip',  action='store_true',
                    help='Skip the process if output_file already exists')
args = parser.parse_args()
#
ddir = args.data_folder
output_file = args.output_file
skip = args.skip
xline = args.xray_line
omode = args.window_mode
sumfile = args.summaries_file
#
if (not os.path.isdir(ddir)):
    print (f'Data folder {ddir} not found.')
    raise FileNotFoundError
if (not os.path.isfile(sumfile)):
    print (f'Summary file {sumfile} not found.')
    raise FileNotFoundError
#
xfold = os.path.basename(ddir)
#
if (skip and os.path.isfile(output_file)):
    print (f"Merged summaries file for {xfold} already exists and skip processing is {skip}. Skipping.")
else:
    t_out = proc_lmon_full(sumfile, inst_mode=omode, xline=xline, data_dir=ddir, output_file=output_file)
    if (t_out is not None):
        t_out.write(output_file,overwrite=True,format='fits')
        print (f"Results for {xfold} for line {xline} and mode {omode} merged")
    else:
        print ('Problems')
#
#
