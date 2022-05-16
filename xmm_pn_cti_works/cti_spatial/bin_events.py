import numpy as np
from astropy.table import Table

def bin_events(events: Table,erange: list[int,int]=(5500,10000),ebin:float=5.0) -> Table:
    '''
    Purpose:
        Bin the input events to create a spectrum
    Inputs:
        events - astropy.Table, the input event list as table, must have column 'PI'
        erange - list, with (PI.min,PI.max) ranges in eV to consider for the binning
        ebin - float, the bin size in eV, default as in XMM-SAS is 5 eV
    Output:
        astropy.Table with two columns: 'bin' and 'counts', wheren 'bin' is the binned centre.

    '''
    bin_edges = np.arange(erange[0],erange[1],ebin)
    hist, bin_edges = np.histogram(events['PI'],bins=bin_edges,range=erange,density=False)
    xmid = (bin_edges[0:-1] + bin_edges[1:])/2.0
    # skip the last one
    table = Table([xmid[0:-1],hist[0:-1]],names=['bin','counts'],dtype=[float,int])
    return table
