#include <cstdio>
#include <cuda_runtime.h>

// naive softmax cpu
void naive_softmax_cpu(const float* A, float* out, int n_rows, int n_cols){
    for (int row=0; row<n_rows; ++row){

        float max_val = A[row*n_cols];
        
        for (int col = 0; col<n_cols; ++col){
            if (A[row * n_cols + col]>max_val){
                max_val = A[row * n_cols + col];
            }
        }

        float sum_exp = 0.0f;
        for (int col=0; col<n_cols; ++col){
            sum_exp += expf(A[row*n_cols+col]- max_val);
        }

        for (int col=0; col < n_cols; ++col){
            out[row*n_cols+col]= expf(A[row*n_cols+col]-max_val)/sum_exp;
        }
    }
}


// naive softmax kernel
__global__ void naive_softmax_cuda(const float* A, float* out, int n_rows, int n_cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_rows && col < n_cols){
        float max_val = -1e20f;
        for (int col_idx=0; col_idx < n_rows; ++col_idx){
            if (A[row * n_cols + col_idx] > max_val) max_val = A[row * 
                    n_cols + col_idx];
        }

        float sum_exp = 0.0f;
        for (int col_idx=0; col_idx< n_cols; ++col_idx){
            sum_exp += expf(in[row* n_cols+col_idx] - max_val);
        }

        out[row * n_cols + col] = expf(A[row * n_cols + col] - max_value)/sum_exp;
    }
}

// launch configuration
int num_rows = 512, num_cols = 1000

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (num_cols + threadsPerBlock.x-1)/threadsPerBlock.x,
    (num_rows + threadsPerBlock.y - 1)/threadsPerBlock.y
);

naive_softmax_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, num_rows, num_cols
);

