#include <cstdio>
#include <cuda_runtime.h>

void tensorAdd3D_cpu(
    const float* A,
    const float* B,
    float* C,
    int depth, 
    int height,
    int width
) {
    for (int d = 0; (d < depth); ++d){
        for (int h = 0; (h < height); ++h){
            for (int w=0; (w < width); ++w){
                int index = d *(height * width)+
                            h * width + w;
                C[index] = A[index] + B[index];
            }
        }
    }
}

//cuda kernel
__global__ tensorAdd3D_cuda(
    const float* A, const float* B,
    float* C, int depth, int height, 
    int width
 ){
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;
    int z_dim = blockIdx.z * blockDim.z + threadIdx.z;

    if (z_dim<depth && y_dim < height && x_dim < width){
        int index = z_dim * (height * width) + y_dim * width + x_dim;
        C[index] = A[index] + B[index];
    }
 }

// launching the cuda kernel
int depth = 32, height = 256, width=256;

dim3 threadsPerBlock(8,8,8);
dim3 blockPerGrid(
    (width + threadsPerBlock.x - 1)/threadsPerBlock.x,
    (height + threadsPerBlock.y -1)/threadsPerBlock.y,
    (depth + threadsPerBlock.z -1)/threadsPerBlock.z
);

tensorAdd3D_cuda<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
    depth, height, width);


//image batch normalization
int batch_size = 64, channels=3, height =128, width=128;

__global__ void normalizeImageBatch(
    const float* input,
    float* output,
    float* mean,
    float* std,
    int batch_size,
    int channels,
    int height,
    int width
) {
    // calculating thread's uique coordinates
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    // apply boundary checks, linearize and perform computation
    if (w < width && h < height && c < channels) {
        for (int b =0; b< batch_size; ++b) {
            int index = b * (channels * height * width)+ 
                        c * (height*width) + 
                        h * width + w;

            output[indx] = (input[iindex] - mean)/std;

        }
    }
}