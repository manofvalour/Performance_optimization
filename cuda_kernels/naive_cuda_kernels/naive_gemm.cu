#include <cstdio>
#include <cuda_runtime.h>

void naive_gemm_cpu(const float* A, const float* B,
                float* C, int M, int N, int K){

    for (int row = 0; row < M; ++row){
        for (int col = 0; col < N; ++col){
            float summation = 0.0f;

            for (int mid = 0; mid < K; ++mid){
                int a_index = row * K + mid;
                int b_index = k * N + col;

                summation += A[a_index] * B[b_index];
            }
            C[row * N + col] = summation;

        }
    }
}


//naive cuda kernel for GEMM
__global__ void naive_gemm_cuda(const float* A,
                        const float* B, float C,
                        int M, int N, int K){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row< M && column< N){
        for (int k_idx = 0; k_idx < K; ++k_idx){
            float sum = 0.0f;
            sum+= A[row * K + k_idx] * B[k_idx * N + column];
        }
        C[row * N + column]=sum;
    }
}


int main(){
    int M = 1024, N = 512, K = 256;

    dim3 threadsPerBlock(16,16);
    dim3 BlockPerGrid(
        (N + threadsPerBlock.x -1)/threadsPerBlock.x,
        (M + threadsPerBlock.y -1)/threadsPerBlock.y
    );

    naive_gemm_cuda<<<BlockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                    M, N, K);
}

// matmul cuda tiled kernel
__global__ void matmul_optimized(const float* A,
    const float* B, float* C, int width){
        __shared__ float As[tile_width][tile_width];
        __shared__ float Bs[tile_width][tile_width];

        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;

        int row = by * tile_width + ty;
        int column = bx * tile_width + tx;

        float accum=0.0f;
        for (int i =0; i<width/tile_width; ++i){
            As[ty][tx] = A[row * width + (i*tile_width+ty)];
            Bs[ty][tx] = B[(i*tile_width + tx) * width + column]
            __syncthreads();

            for (int k=0; k<width; ++k){
                accum += As[ty][k]* B[k][tx];
                __syncthreads();
            
            C[row * width + col] = accum;
            }
        }
    }