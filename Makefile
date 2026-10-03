# Makefile for Parallel Matrix Multiplication
# Usage: 
# module load xl_r spectrum-mpi cuda
# make

NVCC        = nvcc
MPICC       = mpicc
CUDA_ARCH   = -gencode arch=compute_70,code=sm_70
NVCC_FLAGS  = -O3 $(CUDA_ARCH) -Xcompiler "-O3"
MPICC_FLAGS = -O3
CUBLAS_LIBS = -lcublas
MPI_CFLAGS  = $(shell mpicc --showme:compile 2>/dev/null || echo "")
MPI_LDFLAGS = $(shell mpicc --showme:link 2>/dev/null | sed 's/-pthread//g' || echo "")

.PHONY: all clean

# Removed mpi_io_benchmark from here
all: matmul matmul_cpu

matmul: matmul.cu clockcycle.h
	$(NVCC) $(NVCC_FLAGS) \
	    $(foreach flag,$(MPI_CFLAGS),-Xcompiler $(flag)) \
	    matmul.cu -o matmul \
	    $(CUBLAS_LIBS) \
	    $(foreach flag,$(MPI_LDFLAGS),-Xlinker $(flag))

matmul_cpu: matmul_cpu.c clockcycle.h
	$(MPICC) $(MPICC_FLAGS) matmul_cpu.c -o matmul_cpu -lm

clean:
	rm -f matmul matmul_cpu *.bin slurm-*.out