# matmul
Massively parallel matrix multiplication system on the AiMOS supercomputer that goes through a three-way performance comparison: a CPU-only MPI baseline with a cache-friendly triple loop with reordering (i-p-j), a custom tiled shared-memory CUDA kernel, and NVIDIA’s cuBLAS library
