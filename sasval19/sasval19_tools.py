import os
import subprocess
import sys
import logging
import glob

import numpy as np

from astropy.io import fits

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import simple_norm, PercentileInterval, ImageNormalize, ManualInterval
from astropy.nddata import Cutout2D

from regions import CircleSkyRegion

# %%
def run_command_log(command,verbose=True):
    #
    # Execute a shell command with the stdout and stderr being redirected to a log file 
    #
    # shell=False suggested Francesco Pierfederici <fra.pierfederici@icloud.com>, but it does not work as expected
    #
    # not using the other suggestion check=True as I am not sure it will do what I need. It will raise an exception that I am 
    # catching with try: except: anyway.
    #
    # if no logger is configured globally then there will be no saving to a log file
    #
    logger = logging.getLogger(__name__)
    retcode = None
    try:
        result = subprocess.run(command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        retcode=result.returncode
        if retcode < 0:
            print (f"Execution of {command} was terminated by signal", -retcode, file=sys.stderr)
            if (verbose):
                print (f"STDOUT:\n{result.stdout.decode()}")
            logger.error(f"Execution of {command} was terminated by signal: {-retcode} \n {result.stdout.decode()}")
        else:
            print(f"Execution of {command} returned {retcode}", file=sys.stderr)
            if (verbose):
                print (f"STDOUT:\n{result.stdout.decode()}")
            logger.info(f"Execution of {command} was terminated by signal: {-retcode} \n {result.stdout.decode()}")
    except OSError as e:
        print(f"Execution of {command} failed:", e, file=sys.stderr)
        logger.error(f"Execution of {command} failed: {e}")
    return retcode, result

def run_command(command,verbose=True):
    #
    # Execute a shell command with the stdout and stderr being redirected to a log file 
    #
    # shell=False suggested Francesco Pierfederici <fra.pierfederici@icloud.com>, but it does not work as expected
    #
    # not using the other suggestion check=True as I am not sure it will do what I need. It will raise an exception that I am 
    # catching with try: except: anyway.
    #
    retcode = None
    try:
        result = subprocess.run(command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        retcode=result.returncode
        if retcode < 0:
            if (verbose):
                print(f"Execution of {command} was terminated by signal", -retcode, file=sys.stderr)
            logging.warning(f"Execution of {command} was terminated by signal: {-retcode} \n {result.stdout.decode()}")
        else:
            if (verbose):
                print(f"Execution of {command} returned {retcode}", file=sys.stderr)
            logging.info(f"Execution of {command} returned {retcode}, \n {result.stdout.decode()}")
    except OSError as e:
        print(f"Execution of {command} failed:", e, file=sys.stderr)
        logging.error(f"Execution of {command} failed: {e}")
    return retcode, result


# %%
def make_images(event_list,energy_range=(500,2000), bin_size=80, save_prefix="image"):
    #
    # make an image in an energy band [pi0,pi1], FLAG, PATTERN and RAWY linmits as in the pipeline
    #
    # energy_range is in eV
    # bin_size in units of 0.05", default 80 (i.e. 4" pixel)
    #
    # check if event list exists
    #
    if (not os.path.isfile(event_list)):
        logger.error(f'Even list {event_list} does not exist.')
        print(f'Even list {event_list} does not exist.')
        return None
    #
    # read the event list header to get info about the instrument
    #
    with fits.open(event_list) as hdu:
        inst = hdu[0].header['INSTRUME'].strip()
    #
    pi0 = energy_range[0]
    pi1 = energy_range[1]
    # default image binSize is 80 (4"/pix) for PN, because the internal pixels are 0.05"
    #
    image_name = f'{save_prefix}_{pi0}_{pi1}.fits'
    #
    # instrument specific filterting expression
    if ('EMOS' in inst):
        expr = f'(PI in [{pi0}:{pi1}]) && ((FLAG & 0x766ba000)==0) && (PATTERN<=12)'        
    elif ('EPN' in inst):
        expr = f'(PI in [{pi0}:{pi1}]) &&  (RAWY>12) && ((FLAG & 0x2fb0024)==0) && (PATTERN<=4)'
    #
    ev_comm = f'evselect table={event_list} xcolumn=X ycolumn=Y imagebinning=binSize' + \
        f' ximagebinsize={bin_size} yimagebinsize={bin_size}' + \
        f' expression=\'{expr}\' withimageset=true imageset={image_name}'
    #
    status,_ = run_command(ev_comm)
    if (status != 0):
        raise RuntimeError
    #
    print (f"Image in band [{pi0},{pi1}] eV saved to {image_name}") 
    logging.info (f"Image in band [{pi0},{pi1}] eV saved to {image_name}") 
    return image_name


# %%
def make_spectrum(event_list,expression,spectrum_file,do_rmf_arf_grp=False):
    #
    # extract spectrum from `event_list` file filtered with `expression` and saved to `spectrum_file`
    #
    # will also perform arfgen and rmfgen
    #
    # instrument dependency
    with fits.open(event_list) as hdu:
        inst = hdu[0].header['INSTRUME']
    #
    if ('EPN' in inst):
        spec_chan_max = 20479
    elif ('MOS' in inst):
        spec_chan_max = 11999
    else:
        print (f"Cannot extract spectrum for instrument {inst}")
        return None
    #
    # extract the spectrum
    comm = f"evselect table={event_list} withspectrumset=yes spectrumset={spectrum_file}" + \
    f" energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax={spec_chan_max}" +  \
    f" expression='{expression}'"
    #
    print ('Extracting spectrum')
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    if (not do_rmf_arf_grp):
        return
    #
    # now RMF generation
    #
    # now generate the RMF (can take time)
    rmfset = spectrum_file + ".rmf"
    comm = f"rmfgen spectrumset={spectrum_file} rmfset={rmfset}"
    print ('Running rmfgen, can take time...')
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    # now generate the ARF
    arfset = spectrum_file + ".xarf"
    comm = f"arfgen spectrumset={spectrum_file} withrmfset=yes rmfset={rmfset} extendedsource=yes badpixlocation={event_list} detmaptype=flat arfset={arfset}"
    print ('Running arfgen, can take time...')
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    #
    # Use specgroup to add ARF and RMF files in headers. Group to have at least 1 count per bin
    #
    grp0spec = spectrum_file + ".grp"
    comm = f"specgroup spectrumset={spectrum_file} addfilenames=yes rmfset={rmfset}" +         f" arfset={arfset} groupedset={grp0spec} mincounts=1"
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    #
    return


# %%
def make_lightcurve(event_list,expression,rate_file):
    #
    # extract rate curve from `event_list` file filtered with `expression` and saved to `rate_file`
    #
    # extract the lightcurve
    #
    comm = f"evselect table={event_list} withspectrumset=no" +  \
        f" energycolumn=PI withrateset=yes rateset={rate_file} timebinsize=100 maketimecolumn=yes makeratecolumn=yes" + \
        f" expression='{expression}'"
    print ('Extracting rate curve')
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    # apply some corrections
    corr_file = rate_file.replace('.fits','_corr.fits')
    comm = f'epiclccorr srctslist={rate_file} eventlist={event_list} outset={corr_file} withbkgset=no applyabsolutecorrections=no'
    #comm = f'epiclccorr srctslist={rate_file} eventlist={event_list} outset={corr_file} withbkgset=no applyabsolutecorrections=yes'
    print ('Correcting rate curve')
    status,_ = run_command(comm)
    if (status != 0):
        raise RuntimeError
    return corr_file


# %%
def cutout_image(hdu_in,coord,box,out_name="output.fits",clobber=False):
    #
    # Cut an input image in a smaller one, centred on coord (SkyCoord object)
    # with size box (in arcmin) in both x and y direction.
    #
    # Returns the cutout HDU
    #
    #
    zoomSize = u.Quantity((box,box), u.arcmin)
    wcs = WCS(hdu_in[0].header)
    cutout = Cutout2D(hdu_in[0].data, coord, zoomSize, wcs=wcs)
    #
    xdu = fits.PrimaryHDU(cutout.data)
    xdu.header = hdu_in[0].header
    xdu.header.update(cutout.wcs.to_header())
    xdu.writeto(out_name,overwrite=clobber)
    return xdu

