#mpiexec -n 4 --machinefile hosts.txt --map-by node ./proj3 6 53
time mpiexec --mca btl self --mca btl_openib_cpc_include rdmacm --machinefile ./hosts.txt -n 64 --map-by node ./proj3 1000 64
