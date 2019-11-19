#mpiexec -n 4 --machinefile hosts.txt --map-by node ./proj3 6 53
time mpiexec --mca btl self --mca btl_openib_cpc_include rdmacm --machinefile ./hosts.txt -n 4 --map-by node ./a.out 6 53
