#!/bin/sh
#
#$ -V
#$ -l h_vmem=6G
#$ -cwd
#$ -S /bin/bash
#
#$ -M ivan.valtchanov@ext.esa.int
#$ -m ae
#
#$ -e /xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056/results_picorr_mnka/glogs/
#$ -o /xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056/results_picorr_mnka/glogs/
#
echo "Will fit Mn Ka (single line) from events starting rev $rev1"
#
# FF mode, PI
#
python -u $HOME/GitHub/xmmpy/xmm_pn_cti_works/cti_spatial/fit_mnka_picorr.py $rev1 \
     -stacks_dir '/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056' \
     -use_column 'PI' -mode 'FF'
#
# FF mode, PI_CORR
#
#python -u $HOME/GitHub/xmmpy/xmm_pn_cti_works/cti_spatial/fit_mnka_picorr.py $rev1 \
#    -stacks_dir '/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056' \
#    -use_column 'PI_CORR' -mode 'FF'
#
# EFF mode, PI
#
#python -u $HOME/GitHub/xmmpy/xmm_pn_cti_works/cti_spatial/fit_mnka_picorr.py $rev1 \
#    -stacks_dir '/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056' \
#    -use_column 'PI' -mode 'EFF'
#
# EFF mode, PI_CORR
#
#python -u $HOME/GitHub/xmmpy/xmm_pn_cti_works/cti_spatial/fit_mnka_picorr.py $rev1 \
#    -stacks_dir '/xdata/xcaldata/XMM/IVAN/sanders/stacks500_0056' \
#    -use_column 'PI_CORR' -mode 'EFF'
#