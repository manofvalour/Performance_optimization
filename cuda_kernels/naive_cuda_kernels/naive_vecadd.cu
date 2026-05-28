#include <cstdio>
#include <cuda_runtime.h>

// 1D vector add
void vecadd_1d_cpu(float* A, float* B, float* C,
                int M, int N){

    for(int i=0;i<M; ++i){
        for (int j=0; j<N; ++j){
            int index = i*M+j;
            C[index]=A[index]+B[index];
        }
    }
}

__global__ void vecadd_1D_cuda(const float* A, const float* B, float* C,
                        int num_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        C[idx]=A[idx]+B[idx]
    }
}

//2D vector add
void vecdd_2D_cpu(const float* A, const float* B,
                float* C, int n_rows, int n_cols){
    for (i=0;i<n_rows; ++i){
        for (j=0; j<n_cols; ++j){
            int idx = i*n_cols + j;
            C[idx]= A[idx]+B[idx];
        }
    }
}

__global__ void vecadd_2D_cuda(const float* A, const float* B, float* C, int n_rows, int n_cols){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx < n_rows && col_idx < n_cols){
        int index = row_idx*n_cols+col_idx;
        C[index]= A[index]+B[index];
    }
}

// calculating the gridsize, and blocksize
int num_cols = 1000;
int num_rows = 500;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (num_cols * threadsPerBlock.x -1)/threadsPerBlock.x,
    (num_rows * threadsPerBlock.y - 1)/threadsPerBlock.y
);

vecadd_2D_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, num_rows, num_cols);


// 3D add vector
void vecadd_3D_cpu(const float* A, const float* B, float* C,
                int num_rows, int num_cols, int depth){
    for (d=0; d<depth; ++d){
        for (row=0; row<num_rows; ++row){
            for (col=0; col<num_cols; ++col){
                int index = d * (num_rows*num_cols) + row*num_cols + col;
                C[index]= A[index]+B[index];
            }
        }
    }
}

__global__ void vecadd_3D_cuda(const float* A, const float* B, float* C,
                        int n_rows, int n_cols, int depth){
    int idx_x = blockIdx.x* blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx_x<n_rows && idx_y< n_cols && idx_z <depth){
        int index = idx_z *(n_rows*n_cols)+idx_y * n_cols + idx_x;
        C[index]= A[index]+B[index];
    }
}

// calculating the gridsize, and blocksize
int num_cols = 1000;
int num_rows = 500;
int depth = 100

dim3 threadsPerBlock(8,8,8);
dim3 blocksPerGrid(
    (num_cols * threadsPerBlock.x -1)/threadsPerBlock.x,
    (num_rows * threadsPerBlock.y - 1)/threadsPerBlock.y,
    (depth * threadsPerBlock.z - 1)/threadsPerBlock.z
);

vecadd_3D_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, num_rows, num_cols, depth);


