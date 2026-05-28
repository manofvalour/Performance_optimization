#include <cstdio>
#include <cuda_runtime.h>

void naive_conv2D_cpu(const float* in, float* out, const float* kernel,
int height, int width, int kernel_dim){
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;

    for (int row=0; row<output_h; ++row){
        for (int col=0; col<output_width; ++col){
            float sum = 0.0f;
            for (int k_row = 0; k_row<kernel_dim; ++k_row){
                for(int k_col=0; k_row< kernel_dim; ++k_row){
                    int input_row = row + k_row;
                    int input_col = col + k_col;

                    sum += in[input_row * width + input_col] * 
                    kernel[k_row * kernel_dim + k_col];
                }
            }
            out[row * output_w + c] = sum;
        }
    }
}


//naive conv2D cuda kernel
__global__ void naive_conv2D_cuda(const float* in, float* out,
const float* kernel, int height, int width, int kernel_dim){
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    int out_h = height - kernel_dim + 1;
    int out_w = width - kernel_dim + 1;
    
    if (output_row < width && output_col < height){

        float sum = 0.0f;
        for (int k_row=0; k_row < kernel_dim; ++k_row){
            for (int k_col=0; k_col < kernel_dim; ++k_col){
                int input_row = output_row + k_row;
                int input_col = output_col + k_col;
                sum += in[input_row * width + input_col] *
                kernel[k_row * kernel_dim + kernel_col];

            }
        }
        out[input_row * output_w + output_col] = sum;
    }
}

// kernel launch config
int height = 256, width = 256, kernel_dim =3;
int output_h = height - kernel_dim +1;
int output_w = width = kernel_dim + 1;

dim3 threadPerBlock(16,16);
dim3 blocksPerGrid(
    (output_w * threadPerBlock.x-1)/threadPerBlock.x,
    (output_h * threadPerBlock.y - 1)/threadPerBlock.y
);