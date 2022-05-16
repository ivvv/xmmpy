import os
import numpy as np

from astropy.table import Table

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

