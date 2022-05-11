# Works on long-term CTI for XMM-Newton EPIC-pn

Python scripts and notebooks for the full workflow to analyise EPIC-pn data for long-term CTI. 

## Author:

Ivan Valtchanov, XMM SOC, SCO-04, May 2022

##  Package structure:

* `cti_proc`: processing of monitoring resutls as produced by IDL scripts (run by Michael Smith), these are low level scripts to read and combine the results. Containes modules to visualise the residuals.
* `cti_modeling`: scripts (and notebooks) to analyse the results from the monitoring run and model the long-term CTI. Scripts with prefix for each mode: for Small Window (sw), Large Window (lw), Full Frame (ff) and Extended Full Frame (eff) modes. Also contains scripts to generate EPN_CTI_xxxx.CCF files with the derived models.
* `stacking`: scripts to stack observations (in FF and EFF only) in bins of revolutions. Stacked files are used for spatial CTI works.
* `cti_spatial`: scripts to use the stacked event lists, derive spatial offsets in bins of RAWX, RAWY, apply spatial offsets and check the results.
  
