/*
 * CPU-only MPI SUMMA baseline
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "clockcycle.h"

#define PANEL_B 128
#define CLOCK_RATE 512000000ULL

static double cycles_to_sec(uint64_t c) {
    return (double)c / (double)CLOCK_RATE;
}

static void cpu_dgemm(const double *A, const double *B, double *C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            double a = A[i*k + p];
            for (int j = 0; j < n; j++)
                C[i*n + j] += a * B[p*n + j];
        }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 4) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s N proc_rows proc_cols\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int N = atoi(argv[1]);
    int proc_rows = atoi(argv[2]);
    int proc_cols = atoi(argv[3]);

    if (proc_rows * proc_cols != world_size) {
        if (world_rank == 0)
            fprintf(stderr, "proc_rows*proc_cols must equal total ranks\n");
        MPI_Finalize(); return 1;
    }

    int my_row = world_rank / proc_cols;
    int my_col = world_rank % proc_cols;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    int local_rows = N / proc_rows;
    int local_cols = N / proc_cols;

    double *local_A = (double*)malloc(local_rows * N * sizeof(double));
    double *local_B = (double*)malloc(N * local_cols * sizeof(double));
    double *local_C = (double*)calloc(local_rows * local_cols, sizeof(double));
    double *panel_A = (double*)malloc(local_rows * PANEL_B * sizeof(double));
    double *panel_B = (double*)malloc(PANEL_B * local_cols * sizeof(double));

    srand48(world_rank + 1);
    for (int i = 0; i < local_rows * N; i++) local_A[i] = drand48();
    for (int i = 0; i < N * local_cols; i++) local_B[i] = drand48();

    uint64_t t_bcast_a = 0, t_bcast_b = 0, t_compute = 0;
    int steps = (N + PANEL_B - 1) / PANEL_B;

    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t t_start = clock_now();

    for (int step = 0; step < steps; step++) {
        int ps = step * PANEL_B;
        int pw = (ps + PANEL_B <= N) ? PANEL_B : (N - ps);
        int bcast_col_root = step % proc_cols;
        int bcast_row_root = step % proc_rows;

        uint64_t t0 = clock_now();
        if (my_col == bcast_col_root)
            for (int r = 0; r < local_rows; r++)
                memcpy(panel_A + r*pw, local_A + r*N + ps, pw*sizeof(double));
        MPI_Bcast(panel_A, local_rows*pw, MPI_DOUBLE, bcast_col_root, row_comm);
        t_bcast_a += clock_now() - t0;

        t0 = clock_now();
        if (my_row == bcast_row_root)
            memcpy(panel_B, local_B + ps*local_cols, pw*local_cols*sizeof(double));
        MPI_Bcast(panel_B, pw*local_cols, MPI_DOUBLE, bcast_row_root, col_comm);
        t_bcast_b += clock_now() - t0;

        t0 = clock_now();
        cpu_dgemm(panel_A, panel_B, local_C, local_rows, pw, local_cols);
        t_compute += clock_now() - t0;
    }

    uint64_t t_total = clock_now() - t_start;

    uint64_t ga, gb, gc, gt;
    MPI_Reduce(&t_bcast_a, &ga, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_bcast_b, &gb, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &gc, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total,   &gt, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double wall = cycles_to_sec(gt);
        double gflops = (2.0 * (double)N * (double)N * (double)N) / (wall * 1e9);
        printf("=== CPU-only MPI Results ===\n");
        printf("N=%d Ranks=%d Grid=%dx%d\n", N, world_size, proc_rows, proc_cols);
        printf("Wall time : %.4f s\n", wall);
        printf("GFLOPS : %.2f\n", gflops);
        printf("  Bcast A : %.2f%%\n", 100.0*cycles_to_sec(ga)/wall);
        printf("  Bcast B : %.2f%%\n", 100.0*cycles_to_sec(gb)/wall);
        printf("  Compute : %.2f%%\n", 100.0*cycles_to_sec(gc)/wall);
        printf("============================\n");
    }

    free(local_A);
    free(local_B);
    free(local_C);
    free(panel_A);
    free(panel_B);
    MPI_Comm_free(&row_comm); 
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
