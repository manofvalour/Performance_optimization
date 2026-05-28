#include <cstdio>
#include <cuda_runtime.h>

//transposition
void matrix_transpose(const float* in, float* out, int num_rows, int num_cols){
    for (i=0; i<num_rows; ++i){
        for (j=0; j<num_cols; ++j){
            out[j*num_rows+i]=in[i*num_cols+j];
        }
    }
}

__global__ void matrix_transpose_cuda(const float* in, float* out, 
                    int num_rows, int num_cols){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx< num_rows && col_idx < num_cols){
        out[col_idx*num_rows+row_idx]=in[row_idx*num_cols+col_idx];
    }
}

int num_rows = 28;
int num_cols = 28;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (num_cols + threadsPerBlock.x -1)/threadsPerBlock.x,
    (num_rows + threadsPerBlock.y -1)/threadsPerBlock.y
);

for (int batch = 0; batch<100; ++batch){
    float* d_in = device_input + batch *28*28;
    float* d_out = device_output + batch *28*28;

    matrix_transpose_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
}