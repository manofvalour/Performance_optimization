#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE   784
#define HIDDEN_SIZE  256
#define OUTPUT_SIZE  10
#define BATCH_SIZE   8
#define EPOCHS       10
#define LEARNING_RATE 0.001f
#define TRAIN_SIZE   10000

//error handling
#define CUDA_CHECK(call)\
    do{\
        cudaError_t error = call; \
        if (error!= cudaSuccess){\
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//assign Neural Network parameters
typedef struct {
    float *weights1, *weights2, *bias1, *bias2;
    float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2;
    float *grad_output, *dX2;

} NeuralNetwork;

typedef struct {
    double data_loading;
    double fwd_matmul1, fwd_bias1, fwd_relu, fwd_matmul2, fwd_bias2, fwd_softmax;
    double cross_entropy;
    double bwd_output_grad, bwd_matmul1, bwd_matmul2, bwd_bias1, bwd_bias2, bwd_relu, bwd_softmax;
    double weight_updates;
    double total_time;
} TimingStats;

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
}

// Network initialization and memory allocation
void initialize_weights(float *weights, int input_size, int output_size){
    float scale = sqrtf(6.0f/input_size); //Xavier/Glorot initialization
    for (int i= 0; i<input_size * output_size; i++){
        weights[i] = ((float)rand()/RAND_MAX) * 2.0f * scale -scale;
    }
}

void initialize_bias(float *bias, int size){
    for (int i=0; i<size; i++){
        bias[i] = 0.0f; // bias initialized to zero
    }
}

//initializing the neural network parameters on the GPU VRAM
void initialize_neural_network(NeuralNetwork *nn){
    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_output, BATCH_SIZE*OUTPUT_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->dX2, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
   // CUDA_CHECK(cudaMalloc(&nn->grad_input1, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
   // CUDA_CHECK(cudaMalloc(&nn->grad_input2, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
}

//initializing the weight on the CPU DRAM
void initialize_random_weights(NeuralNetwork *nn){
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE*HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

__global__ void matmul(const float *A, const float *B, float *C,
                        int m, int n, int l){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<m && col<n){
        float accum = 0.0f;
        for (int k=0; k<l; ++k){
            accum += A[row * l + k] * B[k*n+col];
        }
        C[row*n+col] = accum;
    }
}


__global__ void bias_sum(float *A, const float *bias, 
                       int batch_size, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b< batch_size && i < size){
        A[idx]+=bias[i];
    }
}

__global__ void relu_forward_kernel(float *x, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size){
    int b = blockIdx.x;
    if (b<batch_size){
        float max_val = x[b*size];
        for (int i =1; i<size; ++i){
            max_val = fmaxf(max_val, x[b*size+i]);
        }

        float sum = 0.0f;
        for (int i =0; i<size; ++i){
            x[b*size+i] = expf(x[b*size+i] - max_val);
            sum+=x[b*size+i];
        }

        for (int i = 0; i<size; ++i){
            x[b*size+i] = fmaxf(x[b*size+i]/sum, 1e-7f);
        }
    }
}


__global__ void matmul_at_b(const float *A, const float *B, float *C,
                        int m, int n, int l){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<m && col<n){
        float accum = 0.0f;
        for (int k=0; k<l; ++k){
            accum += A[k * m + row] * B[k*n+col];
        }
        C[row*n+col] = accum;
    }
}

__global__ void matmul_a_bt(const float *A, const float *B, float *C,
                        int m, int n, int l){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<m && col<n){
        float accum = 0.0f;
        for (int k=0; k<l; ++k){
            accum += A[row * l + k] * B[col*l+k];
        }
        C[row*n+col] = accum;
    }
}


__global__ void compute_output_gradients_kernel(float *grad_output,
                    float *output, int *labels, int batch_size){
    
    int b = blockIdx.x* blockDim.x + threadIdx.x;
    if (b<batch_size){
        for (int i =0; i<OUTPUT_SIZE; ++i){
            grad_output[b*OUTPUT_SIZE+i]= output[b*OUTPUT_SIZE + i];    
        }

        grad_output[b*OUTPUT_SIZE+labels[b]] -= 1.0f;
        
        for (int i = 0; i<OUTPUT_SIZE; ++i){
            grad_output[b*OUTPUT_SIZE+i]/= batch_size;
        }
        
    }  
}

__global__ void relu_backward_kernel(float *grad, float *x, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        grad[idx] *= (x[idx]>0.0f ? 1.0f : 0.0f);
    }
}

__global__ void bias_backward_kernel(float *grad_bias, float *grad,
    int batch_size, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<size){
        float sum = 0.0f;
        for (int b = 0; b<batch_size; ++b){
            sum+= grad[b*size+i];
        }
        grad_bias[i]=sum;
    }
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

// Zero gradients kernel
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// seting up the training loop
void forward_pass(NeuralNetwork *nn, float *input, float *hidden,
float *output, int batch_size, TimingStats *stats){

    struct timespec step_start, step_end;
    
    
    dim3 block_size(32,32);

    dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1)/block_size.x,
                    (batch_size + block_size.y - 1)/block_size.y);

    //launching xw-matmul kernel
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul<<<grid_size1, block_size>>>(input, nn->weights1,
    hidden, batch_size, HIDDEN_SIZE, INPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_matmul1+= get_time_diff(step_start, step_end);

    //launching bias kernel
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_sum<<<(batch_size*HIDDEN_SIZE+255)/256,
    256>>>(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_bias1+= get_time_diff(step_start, step_end);

    //launching relu kernel
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    relu_forward_kernel<<<(batch_size*HIDDEN_SIZE+255)/256,
    256>>>(hidden, batch_size*HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_relu+= get_time_diff(step_start, step_end);

    //implementing second layer(xw+b)
    //running the matmul (xw)
    dim3 grid_size2((OUTPUT_SIZE+block_size.x-1)/block_size.x,
    (batch_size + block_size.y -1)/block_size.y);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul<<<grid_size2, block_size>>>(hidden, nn->weights2,
    output, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_matmul2+= get_time_diff(step_start, step_end);

    //implementing bias
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_sum<<<(batch_size * OUTPUT_SIZE+255)/256, 256>>>(
        output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_bias2+= get_time_diff(step_start, step_end);

    //softmax output
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    softmax_kernel<<<batch_size, 1>>>(output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_softmax+= get_time_diff(step_start, step_end);
}

//backward propagation
void backward_pass(NeuralNetwork *nn, float *input, float *hidden,
float *output, int *labels, int batch_size, TimingStats *stats){
    struct timespec step_start, step_end;
    dim3 block_size(32,32);

    zero_grad_kernel<<<(HIDDEN_SIZE*INPUT_SIZE+255)/256, 256>>>(
        nn->grad_weights1, HIDDEN_SIZE*INPUT_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE*OUTPUT_SIZE+255)/256, 256>>>(
        nn->grad_weights2, HIDDEN_SIZE*OUTPUT_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE+255)/256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE+255)/256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);

    // compute gradient from softmax and cross_entropy
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    compute_output_gradients_kernel<<<(batch_size+255)/256, 256>>>(
        nn->grad_output,output,labels, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_softmax+=get_time_diff(step_start, step_end);

    //layer 2
    dim3 grid_weights2((OUTPUT_SIZE+block_size.x-1)/block_size.x, (
        HIDDEN_SIZE+block_size.y-1)/block_size.y);
    
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_at_b<<<grid_weights2, block_size>>>(hidden, nn->grad_output,
    nn->grad_weights2, HIDDEN_SIZE, OUTPUT_SIZE,batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_matmul2+= get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_backward_kernel<<<(OUTPUT_SIZE+255)/256, 256>>>(nn->grad_bias2,
    nn->grad_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_bias2+=get_time_diff(step_start, step_end);

    //layer 1
    dim3 grid_hidden((HIDDEN_SIZE+block_size.x-1)/block_size.x, 
    (batch_size+block_size.y-1)/block_size.y);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_a_bt<<<grid_hidden, block_size>>>(nn->grad_output, nn->weights2,
    nn->dX2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_output_grad+= get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    relu_backward_kernel<<<(batch_size*HIDDEN_SIZE+255)/256, 256>>>(
        nn->dX2, hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_relu+= get_time_diff(step_start, step_end);
    
    dim3 grid_weights1((HIDDEN_SIZE + block_size.x -1)/block_size.x, (
        INPUT_SIZE + block_size.y - 1)/block_size.y);
    
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_at_b<<<grid_weights1, block_size>>>(input, nn->dX2,
    nn->grad_weights1, INPUT_SIZE, HIDDEN_SIZE, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_matmul1+=get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_backward_kernel<<<(HIDDEN_SIZE+255)/256, 256>>>(
        nn->grad_bias1, nn->dX2, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_bias1+=get_time_diff(step_start, step_end);

 //   CUDA_CHECK(cudaFree(grad_output));
   // CUDA_CHECK(cudaFree(dX2));
  //  CUDA_CHECK(cudaFree(d_ReLU_out));
}


float cross_entropy_loss(float* d_output, int* h_labels, int batch_size){
    
    float h_output[BATCH_SIZE * OUTPUT_SIZE];
    CUDA_CHECK(cudaMemcpy(h_output, d_output,
               batch_size * OUTPUT_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost));
  
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int true_class = h_labels[b];
        float p = h_output[b * OUTPUT_SIZE + true_class];

        // Clamp to [1e-7, 1.0] so log never hits -inf
        if (p < 1e-7f) p = 1e-7f;

        total_loss += -logf(p);
    }

    return total_loss / (float)batch_size;
}

//stochastic gradient descent update
__global__ void sgd_update(float *weights, 
            const float *grads, int size, float lr) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size){
        weights[idx]-= lr * grads[idx];
    }
}

void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec step_start, step_end;

    clock_gettime(CLOCK_MONOTONIC, &step_start);

    sgd_update<<<(INPUT_SIZE*HIDDEN_SIZE+255)/256, 256>>>(
        nn->weights1, nn->grad_weights1, INPUT_SIZE*HIDDEN_SIZE, LEARNING_RATE);

    sgd_update<<<(HIDDEN_SIZE+255)/256, 256>>>(
        nn->bias1, nn->grad_bias1, HIDDEN_SIZE, LEARNING_RATE);

    sgd_update<<<(HIDDEN_SIZE*OUTPUT_SIZE+255)/256, 256>>>(
        nn->weights2, nn->grad_weights2, HIDDEN_SIZE*OUTPUT_SIZE, LEARNING_RATE);
    
    sgd_update<<<(OUTPUT_SIZE+255)/256, 256>>>(
        nn->bias2, nn->grad_bias2, OUTPUT_SIZE, LEARNING_RATE);
    CUDA_CHECK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->weight_updates += get_time_diff(step_start, step_end);
}


// Cuda event timing
/*cudaEvent_ start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start,stop);
print("Kernel time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
*/

// implementing training loop
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train){
    float *d_X_train;
    int *d_y_train;

    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_X_train,
        X_train,
        TRAIN_SIZE * INPUT_SIZE * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_y_train,
        y_train,
        TRAIN_SIZE * sizeof(int),
        cudaMemcpyHostToDevice));

    float *d_hidden, *d_output;
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    int num_batches = TRAIN_SIZE/BATCH_SIZE;

    TimingStats stats = {0}; //initialize a struct

    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++){
            int start_idx = batch * BATCH_SIZE;

            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &d_X_train[start_idx * INPUT_SIZE];
            int *batch_labels_gpu = &d_y_train[start_idx];
            int *batch_labels_cpu = &y_train[start_idx];

            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading+= get_time_diff(step_start, step_end); //create the get_time_diff function
            
            //forward propagation
            forward_pass(nn, batch_input, d_hidden, 
                        d_output, BATCH_SIZE, &stats);
            
            //calculating loss
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(d_output, batch_labels_cpu,
                                            BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start,
                                                    step_end);

            //backward propagation
            backward_pass(nn, batch_input, d_hidden, d_output, 
                        batch_labels_gpu, BATCH_SIZE, &stats);
            
            //Optimizer weight update
            update_weights_timed(nn,&stats);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss/num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    printf("\n === Cuda GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("Detailed Breakdown:\n");
    printf(" Data loading:  %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading/stats.total_time);
    
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf(" Forward_pass: %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);

    printf(" Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy/stats.total_time);
   
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1+stats.bwd_softmax;
    printf(" Backward pass: %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);

    printf(" Weight updates: %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates/stats.total_time);

    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_y_train));
}

//Loading dataset
int read_int_big_endian(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

int load_mnist(float *X, int *y, int max_samples,
               const char *image_path, const char *label_path) {
    
    //open the files
    FILE *img_f = fopen(image_path, "rb");
    FILE *lbl_f = fopen(label_path, "rb");
    
    //check if any of the files failed to open
    if (!img_f || !lbl_f) {
        printf("Could not open MNIST files.\n");
        if (img_f) fclose(img_f);
        if (lbl_f) fclose(lbl_f);
        return 0;
    }

    // Skip headers (magic number, counts, dimensions)
    read_int_big_endian(img_f); // magic
    int n = read_int_big_endian(img_f); //count
    read_int_big_endian(img_f); // rows (28)
    read_int_big_endian(img_f); // cols (28)

    read_int_big_endian(lbl_f); // magic
    read_int_big_endian(lbl_f); // count

    int samples = (n < max_samples) ? n : max_samples;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, img_f);
            X[i * INPUT_SIZE + j] = pixel / 255.0f;  // normalize to [0,1]
        }
        unsigned char label;
        fread(&label, 1, 1, lbl_f);
        y[i] = (int)label;
    }

    fclose(img_f);
    fclose(lbl_f);
    printf("Loaded %d MNIST samples.\n", samples);
    return samples;
}

void free_network(NeuralNetwork *nn){
    cudaFree(nn->weights1);
    cudaFree(nn->weights2);
    cudaFree(nn->bias1);
    cudaFree(nn->bias2);

    cudaFree(nn->grad_weights1);
    cudaFree(nn->grad_weights2);
    cudaFree(nn->grad_bias1);
    cudaFree(nn->grad_bias2);
    cudaFree(nn->grad_output);   // add this
    cudaFree(nn->dX2);  
}

//  main
int main(void) {
    srand(42);  // fixed seed for reproducibility

    // Allocate data buffers
    float *X_train =(float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train =(int*)malloc(TRAIN_SIZE * sizeof(int));
    //float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    //int   *y_train = malloc(TRAIN_SIZE * sizeof(int));
    if (!X_train || !y_train) {
        printf("Out of memory.\n");
        return 1;
    }

    // ── choose data source ────────────────────────────────── 
    int loaded = load_mnist(X_train, y_train, TRAIN_SIZE,
         "data/MNIST/raw/train-images-idx3-ubyte",
         "data/MNIST/raw/train-labels-idx1-ubyte");
    if (!loaded) { free(X_train); free(y_train); return 1; }
    normalize_data(X_train, TRAIN_SIZE*INPUT_SIZE);

    // Init network
    NeuralNetwork nn;
    initialize_neural_network(&nn);
    initialize_random_weights(&nn);

    printf("Network: %d → %d → %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d  |  Epochs: %d  |  LR: %.4f\n\n",
           BATCH_SIZE, EPOCHS, LEARNING_RATE);

    // Train
    train_timed(&nn, X_train, y_train);

    // Final accuracy
    //float acc = compute_accuracy(&nn, X_train, y_train, TRAIN_SIZE);
    //printf("\nTrain accuracy: %.2f%%\n", acc);

    // Cleanup
    free_network(&nn);
    free(X_train);
    free(y_train);

    return 0;
}