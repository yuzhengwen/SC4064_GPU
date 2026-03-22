#define _USE_MATH_DEFINES
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

 /* ============================= PARAMETERS ============================= */

#define C        1.0       // Wave speed
#define DX       0.01      // Grid spacing (dx = dy)
#define DT       0.005     // Time step
#define T_STEPS  200       // Number of time steps to simulate
#define TILE     16        // Shared memory tile dimension

/* Derived: lambda^2 = (c*dt/dx)^2 */
#define LAMBDA2  ((C*DT/DX)*(C*DT/DX))

/* Stability check: lambda <= 1/sqrt(2) ~ 0.707 */
#define LAMBDA   (C*DT/DX)

/* CUDA error checking macro */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

#define CUSPARSE_CHECK(call) do {                                      \
    cusparseStatus_t status = (call);                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",              \
                __FILE__, __LINE__, (int)status);                      \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/* ============================= UTILITIES ============================== */

/*
 * Initialize wave field with u(0,x,y) = sin(pi*x)*sin(pi*y)
 * and u_prev = u_curr (zero initial velocity means u_{-1} = u_0)
 */
void init_wave(double* u_curr, double* u_prev, int N, double dx) {
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        for (int j = 0; j < N; j++) {
            double y = j * dx;
            double val = sin(M_PI * x) * sin(M_PI * y);
            u_curr[i * N + j] = val;
            u_prev[i * N + j] = val;  // du/dt = 0 at t=0
        }
    }
}

/*
 * Save a snapshot of the wave field to binary file for Python visualization.
 * File format: raw double array of N*N values, row-major.
 */
void save_snapshot(const double* u_host, int N, int step) {
    char fname[64];
    snprintf(fname, sizeof(fname), "snapshot_step%04d.bin", step);
    FILE* f = fopen(fname, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); return; }
    fwrite(u_host, sizeof(double), N * N, f);
    fclose(f);
    printf("  Saved snapshot: %s\n", fname);
}

/* Print GPU device info */
void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== GPU: %s ===\n", prop.name);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  Max threads/block: %d\n\n", prop.maxThreadsPerBlock);
}

/* ======================= PART A1: GLOBAL MEMORY ======================= */

/*
 * A1 - Global Memory Stencil Kernel
 *
 * Implements the 5-point stencil directly from global memory.
 * Each thread computes one interior grid point using:
 *   u_next[i,j] = 2*u[i,j] - u_prev[i,j] + lambda^2 * Laplacian(u)[i,j]
 */
// Thread(5, 5) reads: u[4, 5], u[6, 5], u[5, 4], u[5, 6], u[5, 5]
// Thread(5, 6) reads: u[4, 6], u[6, 6], u[5, 5], u[5, 7], u[5, 6] (5,5 repeated)
// repeated memory access (each neighbor read multiple times by adjacent threads).
__global__ void kernel_a1_global(
    double* __restrict__ u_next,
    const double* __restrict__ u_curr,
    const double* __restrict__ u_prev,
    int N, double lambda2)
{
    /* Compute global (i,j) for this thread */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* Enforce Dirichlet BC: boundary nodes stay at 0 */
    if (i == 0 || i >= N - 1 || j == 0 || j >= N - 1) return;

    int idx = i * N + j;

    /* 5-point Laplacian: u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j] */
    double lap = u_curr[(i + 1) * N + j]
        + u_curr[(i - 1) * N + j]
        + u_curr[i * N + (j + 1)]
        + u_curr[i * N + (j - 1)]
        - 4.0 * u_curr[idx];

    /* Time update: leapfrog scheme */
    u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + lambda2 * lap;
}

/* ==================== PART A2: SHARED MEMORY TILED =================== */

/*
 * A2 - Shared Memory Tiled Stencil Kernel
 *
 * Each block loads a (TILE+2) x (TILE+2) tile into shared memory,
 * including a 1-cell halo around the tile boundary. This eliminates
 * redundant global memory reads: each interior cell is read once from
 * global memory and reused by up to 5 neighboring threads.
 *
 * Memory savings vs A1:
 *   A1: Each thread reads 5 values from global memory = 5 * N^2 reads/step
 *   A2: Each tile reads (TILE+2)^2 values = ~(1 + 4/TILE) * N^2 reads/step
 *   Reduction ratio: ~4/TILE per dimension = significant for small tiles
 *
 * Halo loading strategy:
 *   - All threads load their own cell [si, sj]
 *   - Edge threads (tx==0, tx==TILE-1, ty==0, ty==TILE-1) load ghost cells
 *   - Corner threads would need extra loads - handled implicitly
 *   - __syncthreads() ensures all loads complete before stencil computation
 */
__global__ void kernel_a2_shared(
    double* __restrict__ u_next,
    const double* __restrict__ u_curr,
    const double* __restrict__ u_prev,
    int N, double lambda2)
{

    __shared__ double s_curr[TILE + 2][TILE + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.x * TILE + tx;
    int j = blockIdx.y * TILE + ty;
    int si = tx + 1, sj = ty + 1;

    /* Load u_curr tile + halo into shared memory */
    s_curr[si][sj] = (i < N && j < N) ? u_curr[i * N + j] : 0.0;

    if (tx == 0)
        s_curr[0][sj] = (i > 0 && j < N) ? u_curr[(i - 1) * N + j] : 0.0;
    if (tx == TILE - 1)
        s_curr[TILE + 1][sj] = (i < N - 1 && j < N) ? u_curr[(i + 1) * N + j] : 0.0;
    if (ty == 0)
        s_curr[si][0] = (j > 0 && i < N) ? u_curr[i * N + (j - 1)] : 0.0;
    if (ty == TILE - 1)
        s_curr[si][TILE + 1] = (j < N - 1 && i < N) ? u_curr[i * N + (j + 1)] : 0.0;

    __syncthreads();

    if (i == 0 || i >= N - 1 || j == 0 || j >= N - 1) return;
    if (i >= N || j >= N) return;

    double lap = s_curr[si + 1][sj] + s_curr[si - 1][sj]
        + s_curr[si][sj + 1] + s_curr[si][sj - 1]
        - 4.0 * s_curr[si][sj];

    /* u_prev[i,j] is read once per thread — no benefit from shared memory */
    /* u_curr[i,j] is already in s_curr[si][sj] — no extra global read */
    u_next[i * N + j] = 2.0 * s_curr[si][sj] - u_prev[i * N + j] + lambda2 * lap;
}

/* ====================== PART B: cuSPARSE ============================== */

/*
 * Build the discrete 5-point Laplacian matrix L in CSR format on the CPU.
 *
 * For N^2 unknowns (interior + boundary), L is (N^2 x N^2) with:
 *   - Diagonal: -4  (for interior nodes)
 *   - Off-diagonal: +1 for each valid neighbor
 *   - Rows for boundary nodes are identity rows (L[k,k] = 0) to
 *     preserve zero BC via the update formula
 *
 * CSR format: rowPtr[i] to rowPtr[i+1] gives column indices for row i.
 * NNZ (non-zeros): at most 5 per interior row + 1 per boundary row.
 *
 * Note: We use a full N^2 x N^2 matrix (including boundary rows).
 * Boundary rows have a single 0 on the diagonal (value=0), so the
 * time-update formula naturally keeps boundary values at 0.
 */
void build_laplacian_csr(
    int N, int** rowPtr_out, int** colInd_out, double** values_out,
    int* nnz_out)
{
    int N2 = N * N;
    /* Max 5 nonzeros per row */
    int* rowPtr = (int*)malloc((N2 + 1) * sizeof(int));
    int* colInd = (int*)malloc(5 * N2 * sizeof(int));
    double* vals = (double*)malloc(5 * N2 * sizeof(double));

    int nnz = 0;
    rowPtr[0] = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int row = i * N + j;

            /* Boundary node: leave row empty (u_next will remain 0) */
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                /* No entries for boundary rows - u stays 0 via u_next=0 */
                rowPtr[row + 1] = nnz;
                continue;
            }

            /* Interior node: 5-point stencil */
            /* i-1 neighbor */
            colInd[nnz] = (i - 1) * N + j;  vals[nnz++] = 1.0;
            /* j-1 neighbor */
            colInd[nnz] = i * N + (j - 1);  vals[nnz++] = 1.0;
            /* center (diagonal) */
            colInd[nnz] = row;           vals[nnz++] = -4.0;
            /* j+1 neighbor */
            colInd[nnz] = i * N + (j + 1);  vals[nnz++] = 1.0;
            /* i+1 neighbor */
            colInd[nnz] = (i + 1) * N + j;  vals[nnz++] = 1.0;

            rowPtr[row + 1] = nnz;
        }
    }

    *rowPtr_out = rowPtr;
    *colInd_out = colInd;
    *values_out = vals;
    *nnz_out = nnz;
}

/*
 * Small kernel to apply the leapfrog time update after SpMV:
 *   u_next[k] = 2*u_curr[k] - u_prev[k] + lambda2 * Lu[k]
 * Lu is the result of SpMV (L * u_curr), stored in u_next temporarily.
 */
__global__ void kernel_leapfrog(
    double* u_next, const double* u_curr, const double* u_prev,
    int N2, double lambda2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N2) return;
    /* u_next currently holds L*u_curr (from SpMV), apply full update */
    u_next[k] = 2.0 * u_curr[k] - u_prev[k] + lambda2 * u_next[k];
}

/* ====================== SOLVER RUNNER FUNCTIONS ======================= */

/*
 * Run solver using Part A1 (global memory kernel) or A2 (shared memory).
 *   version: 1 = A1 global, 2 = A2 shared
 * Returns: average kernel time per step (ms)
 */
double run_stencil_solver(int N, int version, int save_snaps) {
    size_t bytes = (size_t)N * N * sizeof(double);
    int N2 = N * N;

    /* Host arrays */
    double* h_u = (double*)malloc(bytes);
    double* h_u_prev = (double*)malloc(bytes);
    init_wave(h_u, h_u_prev, N, DX);

    /* Device arrays: u_curr, u_prev, u_next */
    double* d_curr, * d_prev, * d_next;
    CUDA_CHECK(cudaMalloc(&d_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_next, bytes));

    CUDA_CHECK(cudaMemcpy(d_curr, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, bytes));

    /* Grid / block configuration */
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    /* CUDA events for timing */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_kernel_ms = 0.0;

    /* Save initial condition */
    if (save_snaps) {
        CUDA_CHECK(cudaMemcpy(h_u, d_curr, bytes, cudaMemcpyDeviceToHost));
        save_snapshot(h_u, N, 0);
    }

    /* ---- Time stepping loop ---- */
    for (int step = 1; step <= T_STEPS; step++) {

        CUDA_CHECK(cudaEventRecord(ev_start));

        if (version == 1)
            kernel_a1_global << <grid, block >> > (d_next, d_curr, d_prev, N, LAMBDA2);
        else
            kernel_a2_shared << <grid, block >> > (d_next, d_curr, d_prev, N, LAMBDA2);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_kernel_ms += ms;

        /* Rotate buffers: prev <- curr, curr <- next */
        double* tmp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = tmp;

        /* Save snapshots at selected steps */
        if (save_snaps && (step == 50 || step == 100 || step == 200)) {
            CUDA_CHECK(cudaMemcpy(h_u, d_curr, bytes, cudaMemcpyDeviceToHost));
            save_snapshot(h_u, N, step);
        }
    }

    double avg_ms = total_kernel_ms / T_STEPS;

    /* ---- Bandwidth calculation ----
     * 5-point stencil: reads 5 values of u_curr + 1 read of u_prev + 1 write u_next
     * = 7 accesses * 8 bytes (double) = 56 bytes per interior point
     * Simplified formula from assignment spec: 6 * 8 = 48 bytes per grid update */
    long long interior = (long long)(N - 2) * (N - 2);
    double bytes_transferred = interior * 48.0;  /* per assignment spec */
    double bw_GBs = (bytes_transferred / (avg_ms * 1e-3)) / 1e9;

    printf("  [A%d] N=%d | avg kernel: %.4f ms | BW: %.2f GB/s\n",
        version, N, avg_ms, bw_GBs);

    free(h_u); free(h_u_prev);
    cudaFree(d_curr); cudaFree(d_prev); cudaFree(d_next);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    return avg_ms;
}

/*
 * Run solver using Part B (cuSPARSE SpMV).
 * Formulation: u_next = 2*u_curr - u_prev + lambda^2 * L * u_curr
 * Returns: average library call time per step (ms)
 */
double run_cusparse_solver(int N, int save_snaps) {
    int N2 = N * N;
    size_t bytes = (size_t)N2 * sizeof(double);

    /* Host init */
    double* h_u = (double*)malloc(bytes);
    double* h_u_prev = (double*)malloc(bytes);
    init_wave(h_u, h_u_prev, N, DX);

    /* Device wave arrays */
    double* d_curr, * d_prev, * d_next;
    CUDA_CHECK(cudaMalloc(&d_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_next, bytes));

    CUDA_CHECK(cudaMemcpy(d_curr, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev, h_u_prev, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, bytes));

    /* ----- Build Laplacian CSR on CPU, transfer to GPU ----- */
    int* h_rowPtr, * h_colInd; double* h_vals; int nnz;
    build_laplacian_csr(N, &h_rowPtr, &h_colInd, &h_vals, &nnz);

    int* d_rowPtr, * d_colInd; double* d_vals;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (N2 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr, (N2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    /* ----- cuSPARSE setup ----- */
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    /* Create sparse matrix descriptor (CSR) */
    cusparseSpMatDescr_t matL;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matL, N2, N2, nnz,
        d_rowPtr, d_colInd, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    /* Create dense vector descriptors */
    cusparseDnVecDescr_t vecU, vecOut;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecU, N2, d_curr, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecOut, N2, d_next, CUDA_R_64F));

    /* Determine SpMV buffer size */
    double alpha = 1.0, beta = 0.0;
    size_t buf_size;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecU, &beta, vecOut,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));

    void* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_size > 0 ? buf_size : 1));

    /* Leapfrog update grid */
    int block1d = 256;
    int grid1d = (N2 + block1d - 1) / block1d;

    /* CUDA events for timing */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_spmv_ms = 0.0;

    /* ---- Time stepping loop ---- */
    for (int step = 1; step <= T_STEPS; step++) {

        /* Update vecU descriptor to point to current d_curr */
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecU, d_curr));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecOut, d_next));

        /* Time the SpMV call: d_next = 1.0 * L * d_curr + 0.0 * d_next */
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUSPARSE_CHECK(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matL, vecU, &beta, vecOut,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_spmv_ms += ms;

        /* Apply full leapfrog update:
         * d_next = 2*d_curr - d_prev + lambda^2 * d_next (which holds L*u_curr) */
        kernel_leapfrog << <grid1d, block1d >> > (d_next, d_curr, d_prev, N2, LAMBDA2);
        CUDA_CHECK(cudaGetLastError());

        /* Enforce Dirichlet BC (boundary rows in L are zero rows, so d_next
         * boundary elements = 2*0 - 0 + lambda2*0 = 0 automatically) */

         /* Rotate buffers */
        double* tmp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = tmp;

        if (save_snaps && (step == 50 || step == 100 || step == 200)) {
            CUDA_CHECK(cudaMemcpy(h_u, d_curr, bytes, cudaMemcpyDeviceToHost));
            save_snapshot(h_u, N, step);
        }
    }

    double avg_ms = total_spmv_ms / T_STEPS;
    long long interior = (long long)(N - 2) * (N - 2);
    double bytes_spmv = (double)nnz * (sizeof(double) + sizeof(int))
        + (double)N2 * sizeof(double) * 2;
    double bw_GBs = (bytes_spmv / (avg_ms * 1e-3)) / 1e9;

    printf("  [B-cuSPARSE] N=%d | avg SpMV: %.4f ms | eff BW: %.2f GB/s | NNZ: %d\n",
        N, avg_ms, bw_GBs, nnz);

    /* Cleanup */
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecU);
    cusparseDestroyDnVec(vecOut);
    cusparseDestroy(handle);

    cudaFree(d_curr); cudaFree(d_prev); cudaFree(d_next);
    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_vals);
    cudaFree(d_buf);

    free(h_u); free(h_u_prev);
    free(h_rowPtr); free(h_colInd); free(h_vals);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    return avg_ms;
}

/* ======================== MAIN FUNCTION =============================== */

int main(int argc, char** argv) {
    /* Domain scale factor L: N = L / DX */
    double L = 8.0;
    if (argc > 1) L = atof(argv[1]);

    int N = (int)(L / DX) + 1;  /* Grid points per dimension */

    printf("\n================================================\n");
    printf("SC4064 Assignment 2: 2D Wave Equation Solver\n");
    printf("  Domain: [0,%.0f]x[0,%.0f] | N=%d | Steps=%d\n",
        L, L, N, T_STEPS);
    printf("  lambda=%.4f (CFL, must be <= 0.707)\n", LAMBDA);
    printf("  Stability: %s\n", LAMBDA <= 0.7071 ? "OK" : "WARNING: unstable!");
    printf("================================================\n\n");

    print_device_info();

    /* Save snapshots only for the baseline domain (L=1) */
    int save = (L <= 1.01) ? 1 : 0;

    /* ----- Part A1: Global Memory ----- */
    printf("--- Part A1: Global Memory Stencil ---\n");
    cudaEvent_t t0, t1; float total_ms;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    double a1_avg = run_stencil_solver(N, 1, save);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&total_ms, t0, t1);
    printf("  Total A1 time: %.2f ms\n\n", total_ms);

    /* ----- Part A2: Shared Memory ----- */
    printf("--- Part A2: Shared Memory Tiled ---\n");
    cudaEventRecord(t0);
    double a2_avg = run_stencil_solver(N, 2, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&total_ms, t0, t1);
    printf("  Total A2 time: %.2f ms\n\n", total_ms);

    /* ----- Part B: cuSPARSE ----- */
    printf("--- Part B: cuSPARSE SpMV ---\n");
    cudaEventRecord(t0);
    double b_avg = run_cusparse_solver(N, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&total_ms, t0, t1);
    printf("  Total B time: %.2f ms\n\n", total_ms);

    /* ----- Summary ----- */
    printf("=== Summary (N=%d, %d steps) ===\n", N, T_STEPS);
    printf("  A1 avg kernel: %.4f ms\n", a1_avg);
    printf("  A2 avg kernel: %.4f ms\n", a2_avg);
    printf("  B  avg SpMV:   %.4f ms\n", b_avg);
    printf("  A2 speedup over A1: %.2fx\n", a1_avg / a2_avg);
    printf("  A2 speedup over B:  %.2fx\n", b_avg / a2_avg);

    /* Write summary CSV for easy plotting */
    FILE* csv = fopen("perf_summary.csv", "a");
    if (csv) {
        fprintf(csv, "%.0f,%d,%d,%.6f,%.6f,%.6f\n",
            L, N, T_STEPS, a1_avg, a2_avg, b_avg);
        fclose(csv);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
