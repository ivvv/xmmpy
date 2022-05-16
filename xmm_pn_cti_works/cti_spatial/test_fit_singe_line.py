#
# test the code with unbinned event lists
#
from select_stacked_subset import select_stacked_subset
from fit_single_line import fit_single_line

wdir = '/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056/corrected'
#
rev_start = 2000
table = f'{wdir}/FF_stacked_CCD08_{rev_start:04}_{rev_start + 499:04}_0056_corr.fits.gz'
#
tx = select_stacked_subset(table,rawx=[32,33],rawy=[100,120])
#
out = fit_single_line(tx,line_c=8040,erange=[7500,8500],ebin=5,plot_it=True,use_column='PI_CORR')

#out = fit_single_line(tx,line_c=5988,erange=[5500,6500],ebin=5,plot_it=True,use_column='PI_CORR', conf=False)

