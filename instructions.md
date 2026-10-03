1) Log in to AiMOS
2) Copy the code into your AiMOS directory, I copied it into scratch.
3) Before running any code, be sure to run `module load xl_r spectrum-mpi cuda`
4) cd into the directory where the matmul code lives
5) Once in the right directory, where all the code is and where the makefile is, compile everything by running `make`
6) From there, the code can be run. It was specified that all project code will be run using 2 to 4 CPU cores/GPUs, so an example command would look something like the commands below.
7) Confirm the job has finished running using squeue and once the job is done running, then check the slurm-jobID.out file, which should contain the results.
Note: when running MPI I/O checkpointing, when checking slurm file, there should be an extra line of output that looks like -> MPI-IO ckpt: XX%.

For 2 CPU Cores:
```
sbatch -N 1 -n 2 --partition=dcs-2024 --time=10 --wrap="module load spectrum-mpi; mpirun ./matmul_cpu 8192 1 2"
```

For 4 CPU Cores:
```
sbatch -N 1 -n 4 --partition=dcs-2024 --time=10 --wrap="module load spectrum-mpi; mpirun ./matmul_cpu 8192 2 2"
```

For 2 GPUs (this runs cuBLAS, FP64):
```
sbatch -N 1 -n 2 --partition=dcs-2024 --gres=gpu:2 --time=10 --wrap="module load spectrum-mpi cuda; mpirun ./matmul 8192 1 2 1 0"
```

For 4 GPUs (this runs my custom kernel, FP32):
```
sbatch -N 1 -n 4 --partition=dcs-2024 --gres=gpu:4 --time=10 --wrap="module load spectrum-mpi cuda; mpirun ./matmul 8192 2 2 0 1"
```

When running matmul:
mpirun ./matmul N proc_rows proc_cols kernel [prec=0] [ckpt]
- proc_rows and proc_cols represent the total ranks, so for 4 ranks: 1 4, 4 1, or 2 2 would work
- kernel = 0 to custom tilsed kernel, 1 for cuBLAS
- prec = 0 for FP64, 1 for FP32, 2 = TF32, 3 for FP16+TC
- the ckpt is optional and only used if MPI-I/O performance is to be tested and is meant to be a file path

An example of MPI-IO runs:

For 2 GPUs (writing to NVMe):
```
sbatch -N 1 -n 2 --partition=dcs-2024 --gres=gpu:2,nvme:1 --time=10 --wrap="module load spectrum-mpi cuda; export MY_NVME=/mnt/nvme/uid_$UID; mkdir -p \$MY_NVME; mpirun ./matmul 8192 1 2 1 0 /mnt/nvme/uid_\$UID/ckpt_2"
```

For 4 GPUs (writing to home dir):
```
sbatch -N 1 -n 4 --partition=dcs-2024 --gres=gpu:4 --time=10 --wrap="module load spectrum-mpi cuda; mpirun ./matmul 8192 2 2 1 0 \$HOME/ckpt_4"
```

For 4 CPU Cores:
```
sbatch -N 1 -n 4 --partition=dcs-2024 --time=10 --wrap="module load spectrum-mpi; mpirun ./matmul_cpu 8192 2 2 \$HOME/cpu_ckpt"
```