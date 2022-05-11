# In this script, only reading the summary files for each `OBS_ID` and saving it in a table.
# 
# 
# ## History:
# * Created: 08 Oct 2021 based on old notebooks used for the same task, to allow batch processing.
# * Modified it in May 2022, adapted in xmm_pn_cti_works package in cti_proc folder
#     
import os
import argparse 

#from merge_summaries import merge_summaries
from merge_summaries import merge_summaries

home = os.path.expanduser('~')

parser = argparse.ArgumentParser(description='Merge all summary files from monitoring processing for long-term CTI')
parser.add_argument('data_folder', type=str, 
                    help='The absolute path to the root folder for the monitoring results, like "/xdata/xcaldata/XMM/PN/CTI/dat_0062_sci"')
parser.add_argument('merged_summary_file', type=str,
                    help='The name of the merged summaries file')
parser.add_argument('-s','--skip',  action='store_true',
                    help='Skip the process if merged_summary_file already exists')
args = parser.parse_args()
#
ddir = args.data_folder
output_file = args.merged_summary_file
skip = args.skip
#
if (not os.path.isdir(ddir)):
    print (f'Data folder {ddir} not found.')
    raise FileNotFoundError
#
xfold = os.path.basename(ddir)
#
if (skip and os.path.isfile(output_file)):
    print (f"Merged summaries file for {xfold} already exists and skip processing is {skip}. Skipping.")
else:
    df_out = merge_summaries(ddir)
    df_out.to_csv(output_file)
    print (f"Summaries for {xfold} merged")
#
#
