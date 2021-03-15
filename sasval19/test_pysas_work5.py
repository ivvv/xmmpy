import os
from pysas.wrapper import Wrapper

home = os.path.expanduser('~')
#
# will use folder in SAS validation archive with already available event lists
#
svdata = '/xdata/sasval/archive/product/SAS_19.0.0_GNU_CC_CXX_9.2.0/xmmsas_20200825_1931/0060_0122700101'
#
evlist = f'{svdata}/0060_0122700101_EPN_S001_ImagingEvts.ds'
ccf = f'{svdata}/ccf'
os.environ['SAS_CCF'] = ccf
#
#
# now, let's make a spectrum
#
spec_out = 'spectrum.ds'
spec_chan_max = 20479
#
#my_args2 = [f'table={evlist}','withspectrumset=yes', f'spectrumset={spec_out}','energycolumn=PI', 'spectralbinsize=5',\
#    'specchannelmin=0', f'specchannelmax={spec_chan_max}']

my_args2 = [f'table={evlist}','withspectrumset=yes', f'spectrumset={spec_out}','energycolumn=PI', 'spectralbinsize=5',\
    'withspecranges=yes','specchannelmin=0',f'specchannelmax={spec_chan_max}']

#my_args2 = [f'table={evlist}','withspectrumset=yes', f'spectrumset={spec_out}','energycolumn=PI', 'spectralbinsize=5',\
#    'specchannelmin=0',f'specchannelmax={spec_chan_max}']

p = Wrapper('evselect',my_args2)

print ('evselect',' '.join(my_args2))

p.run()
#
# now make an image in band [0.5:2] keV
#
pi0 = 500
pi1 = 2000
image_out = f'image_{pi0}_{pi1}.ds'
my_args3 = [f'table={evlist}','withimageset=yes', f'imageset={image_out}','energycolumn=PI',\
    'expression=' + f'(PI in [{pi0}:{pi1}]) &&  (RAWY>12) && ((FLAG & 0x2fb0024)==0) && (PATTERN<=4)']

p2 = Wrapper('evselect',my_args3)

print ('evselect',' '.join(my_args3))

p2.run()
