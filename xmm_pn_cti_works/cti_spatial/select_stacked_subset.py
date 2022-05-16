import os
from astropy.table import Table


def select_stacked_subset(filename: str, rawx: list=(1,64), rawy: list=(12,200), erange: list=(5500,10000), verbose: bool=True) -> Table:
    '''
    PURPOSE:
        From a stacked file (per CCD), select subsets of events with RAWX, RAWY and energy range

    '''
    if (not os.path.isfile(filename)):
        print (f'Error! cannot find input file {filename}. Cannot continue.')
        return None
    #
    events = Table.read(filename)
    #
    nin = len(events)
    # first, filter on PI
    xmask = (events['PI'] >= erange[0]) & (events['PI'] < erange[1])
    tmp = events[xmask]
    n1 = len(tmp)
    if (n1 < 1):
        print (f'Warning! No events selected with PI in [{erange[0]},{erange[1]})')
        return None
    # next, filter spatially on RAWX and RAWY
    xmask = (tmp['RAWX'] >= rawx[0]) & (tmp['RAWX'] < rawx[1]) & \
        (tmp['RAWY'] >= rawy[0]) & (tmp['RAWY'] < rawy[1])
    #
    tout = tmp[xmask]
    n2 = len(tout)
    if (n2 < 1):
        print (f'Warning! No events selected with RAWX in [{rawx[0]},{rawx[1]}) and RAWY in [{rawy[0]},{rawy[1]})')
        return None
    if (verbose):
        print (f'Selected a subset of events, n={n2} out of total number of input events {nin}')
    return tout



