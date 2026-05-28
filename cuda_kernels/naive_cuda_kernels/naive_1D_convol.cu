#include <cstdio>
#include <cuda_runtime.h>

void naive_conv1D_cpu(const float* in, float* out, 
const float* kernel,int input_size, int kernel_size){
    int output_size = input_size - kernel_size + 1;

    for (int i=0; i< output_size; ++i){
        float sum = 0.0f;

        for (int j = 0; j< kernel_size; ++j){
            sum+= in[i+j] * kernel[j];
        }
        out[i] = sum;
    }
}

// naive conve1D cuda kernel
__global__ void naive_conv1D_cuda(const float* in, float* out,
const float* kernel, int input_size, int kernel_size){
    int output_idx = blockIdx.x* blockDim.x+ threadIdx.x;
    int output_size = input_size - kernel_size +1;

    if (output_idx< output_size){
        float sum = 0.0f;

        for (int i=0; i<kernel_size; ++i){
            sum+= in[output_idx+i]* kernel[i];
        }
        out[output_idx]= sum;
    }
}

// launch configuration
void main(void){
    int input_size = 1000000, kernel_size = 32;
    int output_size = input_size - kernel_size +1;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x -1)/threadsPerBlock.x
    );

    naive_conv1D_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        d_in, d_out, d_kernel, input_size, kernel_size
    );
}
