import os
import glob
import numpy as np

from astropy.table import Table, vstack

from . import extract_events, bin_events

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
