#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

template <class T>
__global__ void

spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{
    // Identify the row index for the current thread block
    int row = blockIdx.x;
    // Identify the thread ID within the block
    int thread_id = threadIdx.x;
    // Determine the number of threads per row
    int threads_per_row = blockDim.x;
    // Initialize a variable sum to accumulate the dot product for the current row
    double sum = 0.0;
    // Iterate through the non-zero elements of the ELL matrix using thread parallelism
    for (int i = 0; i < nnz; i += threads_per_row) {
        int col = col_ind[i + thread_id];
        if (col < n) {
            sum += vals[i + thread_id] * x[col];
        }
    }
    // Shared memory to store partial sums for reduction
    __shared__ double shared_sum[64];
    shared_sum[thread_id] = sum;
    // Perform parallel reduction to obtain the final dot product for the row
    for (int stride = threads_per_row / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
    }
    // Store the final result in the output vector 'b'
    if (thread_id == 0) {
        b[row] = shared_sum[0];
    }
}


void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    unsigned int blocks = m; 
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);

    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}


void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n,
                      int nnz, double* x, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // Allocate device memory for ELL matrix column indices, values, input vector, and output vector
    checkCudaErrors(cudaMalloc((void**)dev_col_ind, nnz * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)dev_vals, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)dev_x, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)dev_b, m * sizeof(double)));

    // Copy host data to device memory for ELL matrix column indices, values, and input vector
    checkCudaErrors(cudaMemcpy(*dev_col_ind, col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*dev_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize the output vector on the device to zero
    checkCudaErrors(cudaMemset(*dev_b, 0, m * sizeof(double)));
}


void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // Allocate device memory for CSR matrix row pointers, column indices, values, input vector, and output vector
    checkCudaErrors(cudaMalloc((void**)dev_row_ptr, (m + 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)dev_col_ind, nnz * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)dev_vals, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)dev_x, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)dev_b, m * sizeof(double)));

    // Copy host data to device memory for CSR matrix row pointers, column indices, values, and input vector
    checkCudaErrors(cudaMemcpy(*dev_row_ptr, row_ptr, (m + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*dev_col_ind, col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*dev_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize the output vector on the device to zero
    checkCudaErrors(cudaMemset(*dev_b, 0, m * sizeof(double)));
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
                             int m, int n, int nnz, double* x, double* b)
{
    // Identify the row index for the current thread block
    unsigned int row = blockIdx.x;
    // Identify the start and end indices for the non-zero elements of the current row
    unsigned int start = row_ptr[row];
    unsigned int end = row_ptr[row + 1];
    // Initialize a variable sum to accumulate the dot product for the current row
    T sum = 0.0;
    // Perform the dot product using thread parallelism
    for (unsigned int i = start + threadIdx.x; i < end; i += blockDim.x) {
        sum += vals[i] * x[col_ind[i]];
    }

    // Perform parallel reduction within a warp to obtain the final dot product for the row
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

   // Use shared memory to store partial sums for inter-warp reduction
    __shared__ T shared_sum[64];
    if (threadIdx.x % warpSize == 0) {
        shared_sum[threadIdx.x / warpSize] = sum;
    }

    __syncthreads();

    // Perform inter-warp reduction to obtain the final dot product for the row
    if (threadIdx.x < warpSize) {
        sum = (threadIdx.x < blockDim.x / warpSize) ? shared_sum[threadIdx.x] : 0.0;

        for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
    }

   // Store the final result in the output vector 'b'
    if (threadIdx.x == 0) {
        b[row] = sum;
    }
}

void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

     //GPU execution parameters
     //1 thread block per row
     //64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




