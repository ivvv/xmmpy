#
# fitting Mn Ka on grid
#
qsub -N ff_mnka_0000 -v rev1=0 ./run_fit_on_grid.sh
qsub -N ff_mnka_0250 -v rev1=250 ./run_fit_on_grid.sh
qsub -N ff_mnka_0500 -v rev1=500 ./run_fit_on_grid.sh
qsub -N ff_mnka_0750 -v rev1=750 ./run_fit_on_grid.sh
#qsub -N ff_mnka_1000 -v rev1=1000 ./run_fit_on_grid.sh
#
# qsub -N ff_mnka_1250 -v rev1=1250 ./run_fit_on_grid.sh
# qsub -N ff_mnka_1500 -v rev1=1500 ./run_fit_on_grid.sh
# qsub -N ff_mnka_1750 -v rev1=1750 ./run_fit_on_grid.sh
# qsub -N ff_mnka_2000 -v rev1=2000 ./run_fit_on_grid.sh
# #
# qsub -N ff_mnka_2250 -v rev1=2250 ./run_fit_on_grid.sh
# qsub -N ff_mnka_2500 -v rev1=2500 ./run_fit_on_grid.sh
# qsub -N ff_mnka_2750 -v rev1=2750 ./run_fit_on_grid.sh
# qsub -N ff_mnka_3000 -v rev1=3000 ./run_fit_on_grid.sh
# #
# qsub -N ff_mnka_3250 -v rev1=3250 ./run_fit_on_grid.sh
# qsub -N ff_mnka_3500 -v rev1=3500 ./run_fit_on_grid.sh
# qsub -N ff_mnka_3750 -v rev1=3750 ./run_fit_on_grid.sh
#qsub -N ff500_st_4000 -v rev1=4000 ./run_fit_on_grid.sh
