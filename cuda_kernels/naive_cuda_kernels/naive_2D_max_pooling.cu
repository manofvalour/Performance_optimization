#include <cstdio>
#include <cuda_runtime.h>

// naive cpu max pooling function
void 2D_maxpool_cpu(const float* in, float* out, 
int M, int N, int pool_dim){

    int output_h = M / pool_dim;
    int output_w = N / pool_dim;

    for (int row=0; row<output_h; ++row){
        for (int col=0; col< output_w; ++col){
            float max_val = -1e20f;
            for  (int pool_r=0; pool_r<pool_dim; ++pool_r){
                for (int pool_c = 0; pool_c<pool_dim; ++pool_c){
                    int input_row = row * pool_dim + pool_r;
                    int input_col = col * pool_dim + pool_c;

                    float val = in[input_row * width + input_col];
                    if (val > max_val) max_val = val;
                }
            }
            out[row * output_w + col] = max_val;
        }
    }
}

// implementing naive cuda kernel
__global__ void 2D_maxpool_cuda(const float* in, float* out,
int M, int N, int pool_dim){
    int output_h = M/pool_dim;
    int output_w = N/pool_dim;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<output_h && col<output_w){
        float max_val = -1e20f;
        for (int pool_r = 0; pool_r < pool_dim; ++pool_r){
            for (int pool_c = 0; pool_c < pool_dim; ++pool_c){
                int input_row = row * pool_dim + pool_row;
                int input_col = col * poo_dim + pool_col;

                float val = in[input_row * N + input_col];
                if (val>max_val) max_val = val;
            }
        }
        out[row * output_w + col] = max_val;
    }
}


// launch configuration
int height = 256, width = 256, pool_dim = 2;
int output_h = height/pool_dim;
int output_w = width/pool_dim;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (output_w + threadsPerBlock.x-1)/threadsPerBlock.x,
    (output_h + threadsPerBlock.y-1)/threadsPerBlock.y
);

2D_maxpool_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, height, width, pool_dim
);