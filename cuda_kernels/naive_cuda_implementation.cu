#include <cstdio>
#include <cuda_runtime.h>

//vector add
//naive 1D vector add
void 1D_vec_add_cpu(const float* A, const float* B, float* calculating
int n){
    for (int i = 0; i<n; ++i){
        C[i] = A[i] + B[i];
    }
}

// naive 1D vector add
__global__ void 1D_vec_add_cuda(const float* A, const float* B,
float* C, int n){
    int in = blockIdx.x * blockDim.x + threadIdx.x;
    if (in<n){
        c[in] = A[in] + B[in];
    }
}

//launch configuration
int n = 1000;
dim3 threadsPerBlock(256);
dim3 blocksPerGrid((n+threadsPerBlock.x-1)/threadsPerBlock.x);

1D_vec_add_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, n
);

// naive 2dvector add
void 2D_vec_add_cpu(const float* A, const float* B, float* C,
int M, int N){
    for (int i = 0; i<M; ++i){
        for (int j = 0; i<N; ++j){
            int index = i*N + j;

            C[index] = A[index] + B[index];
        }
    }
}

//naive 2dvector add cuda
__global__ void 2D_vec_add_cuda(const float* A, const float* B, float* C,
int M, int N){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N){
        int index = i * N + j;
        C[index] = A[indx] + B[index];
    }
}

//launch configuration
int n_row = 1000, n_col=700;
dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (n_col+threadsPerBlock.x-1)/threadsPerBlock.x,
    (n_row+threadsPerBlock.y-1)/threadsPerBlock.y
);

2D_vec_add_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, n_row, n_col
);


// naive 3D vector add
void naive_3D_vec_add_cpu(const float* A, const float* B, float* C,
int height, int width, int depth){
    for (int row = 0; row<height; ++row){
        for(int col = 0; col<width; ++col){
            for (int de = 0; de<depth; ++de){
                int index = de * (height*width) + row * width + col;
                C[index] = A[index] + B[index];
            }
        }
    }
}

// naive 3D vector add cuda
__global__ void naive_3D_vec_add_cuda(const float* A, const float* B, float* C, 
int height, int width, int depth){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dept_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (row_idx < height && col_idx < width && dept_idx < depth){
    
        int index = dept_idx * (height*width) + row_idx * width + col_idx;
        C[index] = A[index] + B[index];
    }
}

//launch configuration
int n_row = 1000, n_col=700, depth = 256;
dim3 threadsPerBlock(8,8,8);
dim3 blocksPerGrid(
    (n_row+threadsPerBlock.y-1)/threadsPerBlock.y,
    (n_col+threadsPerBlock.x-1)/threadsPerBlock.x,
    (depth+ threadsPerBlock.z-1)/threadsPerBlock.z
);

3D_vec_add_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, n_row, n_col, depth
);

//matrix multiplication (GEMM)
//2d matmul cpu
void 2D_GEMM_cpu(const float* A, const float* B, 
float* C, int M, int N, int K){

    for (int row=0; row< M; ++row){
        for (int col = 0; col < N; ++col){
    
            float accum = 0.0f;
            for (int k_idx = 0; k_idx < K; ++k_idx){
    
                int idx_a = row * K + k_idx;
                int idx_b = k_idx * N + col;

                accum += A[idx_a] * B[idx_b];
            }
            C[row * N + col] = accum;
        }
    }
}

//2d matmul cuda
__global__ void 2D_GEMM_cuda(const float* A, const float* B, 
float* C, int M, int N, int K){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx<M && col_idx<N){
        float accum = 0.0f;

        for (int k_idx=0; k_idx<K; ++k_idx){
            int index_a = row_idx * K + k_idx;
            int index_b = k_idx * N + col_idx;

            accum += A[index_a]* B[index_b] 
        }
        C[row_idx * N + col_idx] = accum;
    }

}

//launch configuration
int M = 1000, N = 600, K = 316;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (N + threadsPerBlock.x-1)/threadsPerBlock.x,
    (M + threadsPerBlock.y-1)/threadsPerBlock.y
);

2D_GEMM_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, M, N, K
);

//3d matmul cpu
void 3D_GEMM_cpu(const float* A, const float* B, 
float* c, int M, int N, int K, int depth){
    for (int d=0; d<depth; ++d){
        for (int row=0; row<M; ++row){
            for (int col=0; col<N; ++col){
                float accum = 0.0f;

                for (int k_idx=0; k_idx<K; ++k_idx){
                    accum+= A[d*(M*K)+ row*K+ k_idx] * 
                            B[d*(K*N)+ k_idx*N + col];
                }
                C[d*(M*N)+row*N+col] = accum;
            }
        }
    }
}

//3dmatmul cuda
__global__ void 3D_GEMM_cuda(const float* A, const float*B,
float C, int M, int N, int depth, int K_share){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx< M && col_idx< N){
        for (int d=0; d<depth; ++d){
            float accum = 0.0f;
            for (int k_idx =0; k_idx<K; ++k_idx){
                accum+= A[d*(M*K)+ row_idx*K+k_idx] * 
                        B[d*(K*N) + k_idx*N+col_idx];
            }
            C[d*(M*N)+row_idx*N+col_idx];
        }
    }
}

//launch configuration
int M=1000, N=699, K = 512, depth = 256;
dim3 threadsPerBlock(8,8,8);
dim3 blocksPerGrid(
    (N+threadsPerBlock.x-1)/threadsPerBlock.x,
    (M+threadsPerBlock.y-1)/threadsPerBlock.y,
    (depth+threadsPerBlock.z-1)/threadsPerBlock.z
)

3D_GEMM_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_A, d_B, d_C, M, N, K, depth
)

//matrix_transpose
//2d matrix transpose cpu
void 2D_transpose_cpu(const float* in, float* out,
int M, int N){
    for (int row=0; row<M; ++row){
        for (int col=0; col<N; ++col){
            out[col * M + row] = in[row * N + col]; 
        }
    }
}
//2d matrix transpose gpu
__global__ void 2D_transpose_cuda(const float* in,
float* out, int M, int N){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx<M && col_idx<N){
        out[col_idx * M + row_idx] = in[row_idx * N + col_idx];
    }
}
//launch configuration
int M=128, N = 128, batch_size = 100;
dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (N+threadsPerBlock.y-1)/threadsPerBlock.y,
    (M+threadsPerBlock.x-1)/threadsPerBlock.x
);

for (int batch=0; batch< batch_size; ++batch){
    int d_in = device_input + batch * M * N;
    int d_out = device_output + batch * M * N;
}


2D_transpose_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, M, N
);

//3d matrix transpose cpu
void 3D_transpose_cpu(const float* in, float* out,
int M, int N , int depth){
    for (int d = 0; d<depth; ++d){
        for (int row=0; row<M; ++row){
            for (int col=0; col<N; ++col){
                out[d*(M*N)+ col * M + row] = in[d*(M*N)+row*N+col];
            }
        }
    }
}

//3d matrix transpose gpu
__global__ void 3D_transpose_cuda(const float* in, float* out,
int M, int N, int depth){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && col_idx < N){
        for (int d=0; d<depth; ++d){
            out[d*(M*N)+col_idx*M+row_idx] = in[d*(M*N)+row_idx*N+col_idx];
        }
    } 
}

//launch configuration
int M = 28, N=28, batch_size=100, depth = 3;

dim3 threadsPerBlock(8,8,8);
dim3 blocksPerGrid(
    (N*threadsPerBlock.x+1)/threadsPerBlock.x,
    (M* threadsPerBlock.y+1)/threadsPerBlock.y,
    (Z* threadsPerBlock.z+1)/threadsPerBlock.z
);

for (int block=0; block<block_size; ++block){
    int d_in = device_input + block * M*N;
    int d_out = device_output + block * M*N;

    3D_transpose_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        d_in, d_out, M, N, depth
    );

}

//softmax
//softmax 2d cpu
void 2D_softmax_cpu(const float* in, float* out, int M, int N){
    for (int row=0; row<M; ++row){

        float max_val = in[row*N];
        for (int col=0; col<N; ++col){
            val = in[row*N+col];
            if (val>max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int col=0; col<N; ++col){
            sum+= expf(in[row*N+col]- max_val);
        }

        for (int col = 0; col<N; ++col){
            out[row*N+col]= expf(in[row*N+col]-max_val)/sum;
        }
    }
}

//softmax 2d cuda
__global__ void 2D_softmax_cuda(const float* in, float* out,
M, N){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx<M && col_idx < N){

        float max_val = -1e20f;
        for (int col=0; col<N; ++col){
            val = in[row_idx*N+col]
            if (val>max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int col=0; col<N; ++col){
            sum+= expf(in[row_idx*N+col]-max_val);
        }

        out[row_idx*N+col_idx] = expf(in[row_idx*N+col_idx])/sum;
    }
}

//launch configuration
int M=1000, N=500;
dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (N*threadsPerBlock.x -1)/threadsPerBlock.x,
    (M*threadsPerBlock.y - 1)/threadsPerBlock.y
);

2D_softmax_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in,d_out, M, N
)

//softmax 3d cpu
void 3D_softmax_cpu(const float* in, float* out, int M,
int N, int depth){
    for (int d=0; d<depth; ++d){
        for (int row=0; row<M; ++row){
            float max_val = -1e20f;
            for (int col=0; col<N; ++col){
                val = in[row*N+col];
                if (val>max_val) max_val=val;
            }
            float sum = 0.0f;
            for (int col=0; col<N; ++col){
                sum+= expf(in[row*N+col]-max_val);
            }

            for (int col=0; col<N; ++col){
                out[row*N+col]= expf(in[row*N+col]/sum);
            }
        }
    }
}

//softmax 3d cuda
__global__ void 3D_softmax_cuda(const float* in, float* out, int M,
int N, int depth){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (row_idx<M && col_idx<N){
        float max_value = -1e20f;
        for (int col = 0; col<N; ++col){
            val = in[dept_idx*(M*N)+row_idx*N+col];
            if (val>max_val) max_val = val;
        }
        

        float sum = 0.0f;
        for (int col = 0; col<N; ++col){
            sum+= expf(in[row_idx*N+col]-max_value);
        }

        out[d*(M*N)+ row_idx*N + col_idx]= expf(in[d*(M*N)+row_idx*N + col_idx] - max_value)/sum;
    }
}

//launch configuration
int M=1000, N=512, depth = 256;
dim3 threadsPerBlock(8,8,8);
dim3 blocksPerGrid(
    (N+threadsPerBlock.x-1)/threadsPerBlock.x,
    (M+threadsPerBlock.y-1)/threadsPerBlock.y,
    (depth+threadsPerBlock.z-1)/threadsPerBlock.z
);

3D_softmax_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, M, N, depth
)

//1D_conv
void 1D_conv_cpu(const float* in, float* out, 
const float* kernel, int input_size, int kernel_size){
    int output_size = input_size - kernel_size +1;

    float sum = 0.0f;
    for (int i = 0; i<output_size; ++i){
        for (int k =0; k<kernel_size; ++k){
            sum+= in[i+k]*kernel[k];
        }
        out[i]=sum;
    }
}

// 1D conv cuda kernel
__global__ void 1D_convo_cuda(const float* in, float* out,
const float* kernel, int in_size, int k_size){
    int idx = blockIdx.x * blockDim.x + threadsPerBlock.x;
    int output_size = in_size - k_size +1;

    if (idx< output_size){
        float sum = 0.0f;
        for (int k=0; i<kernel_size; ++k){
            sum+= in[idx+k]* kernel[k];
        }
        out[idx]= sum;
    }
}

//launch configuration
int input_size = 1000, kernel_size = 5

output_size = input_size - kernel_size +1;

dim3 threadsPerBlock(32);
dim3 blocksPerGrid(
    (output_size+threadsPerBlock.x-1)/threadsPerBlock.x
);

1D_convo_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, d_kernel, in_dim, k_dim
)

//2D_conv
//2D conv cpu
void conv2D_cpu(const float* in, float* out, 
const float* kernel, int M, int N, int kernel_size){
    int output_M = M - kernel_size + 1;
    int output_N = N - kernel_size + 1;

    for (int i=0; i<output_M; ++i){
        for (int j=0; j<output_N; ++j){
            float sum = 0.0f;
            for (int k_row =0; k_row< kernel_size; ++k_row){
                for (int k_col=0; k_col<kernel_size; ++k_col){
                    int input_row = i+k_row;
                    int input_col = j + k_col;

                    sum += in[input_row * M + input_col] * 
                            kernel[k_row * kernel_size + k_col];
                }
            }
            out[i * ouput_N + j] = sum;
        }
    }
}

//2d conv cuda
__global__ void conv2D_cuda(const float* in, 
float* out, const float* kernel, int M, 
int N, int kernel_dim){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_h = M - kernel_dim + 1;
    int output_w = N - kernel_dim + 1;

    if (row_idx< M && col_idx<N){
        int sum = 0.0f;
        for (int k_row=0; k_row<kernel_dim; ++k_row){
            for (int k_col=0; k_col<kernel_dim; ++k_col){
                int input_row = row_idx * k_row;
                int input_col = col_idx * k_col;

                sum += in[input_row * M * input_col] * 
                    kernel[k_row* kernel_dim + k_col]; 
            }
        }
        out[row_idx * outpu_w + col_idx] = sum;
    }

}

//launch configuration
int h = 1000, w=500, k_dim = 5;

int output_h = h - k_dim + 1;
int output_w = w - k_dim + 1;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (output_w * threadsPerBlock.x - 1)/threadsPerBlock.x,
    (output_h * threadsPerBlock.y - 1)/threadsPerBlock.y
);

conv2D_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, kernel, h, w, k_dim
)


//2D_maxpool
//2D maxpool cpu
void maxpool2D_cpu(const float* in, float* out,
const float* pool, int height, int width, int pool_dim){
    int out_h = height - pool_dim + 1;
    int out_w = width - pool_dim + 1;

    for (int i =0; i<out_h; ++i){
        for (int j=0; j<out_w; ++j){
            int max_pool = -1e20f;
            for (int k_row =0; k_row<pool_dim; ++k_row){
                for (int k_col = 0; k_col< pool_dim; ++k_col){
                    int input_row = i * k_row;
                    int input_col = j * k_col;

                    if (in[input_row * pool_dim + input_col]>max_pool) max_pool=int[input_row * pool_dim + input_col];

                }
            }
            out[i* out_w + j] = max_pool;
        }
    }
}

//2D maxpool cuda
__global__ void pool2D_cuda(const float* in, float* out,
const float* kernel, int height, int width, int pool_dim){
    int out_h = height - pool_dim + 1;
    int out_w = width - pool_dim + 1;

    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_h < row_idx && out_w < col_idx){

        float max_pool = -1e20f;
        for (int k_row =0; k_row< pool_dim; ++k_row){
            for (int k_col = 0; k_col < pool_dim; ++k_col){
                int input_row = row_idx * pool_dim;
                int input_col = col_idx * pool_dim;

                float val = in[input_row * pool_dim + input_col];
                if (val>max_pool) max_pool = val;
            }
        }
        out[row_idx * out_w + col_idx] = max_pool;
    }
}

//launch configuration
int height = 1000, width =500, pool_dim = 5;

int out_h = height - pool_dim + 1;
int out_w = width - pool_dim + 1;

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid(
    (out_w + threadsPerBlock.x - 1)/threadsPerBlock.x
    (out_h + threadsPerBlock.y - 1)/threadsPerBlock.y
);

pool2D_cuda<<<blocksPerGrid, threadsPerBlock>>>(
    d_in, d_out, kernel, height, width, pool_dim
)
