#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE   784
#define HIDDEN_SIZE  256
#define OUTPUT_SIZE  10
#define BATCH_SIZE   8
#define EPOCHS 10
#define LEARNING_RATE 0.001f
#define TRAIN_SIZE   10000

//cuda error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess){ \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//cublas error handling
#define CUBLAS_CHECK(call) \
    do{\
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) {\
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);\
            exit(EXIT_FAILURE);\
        } \
    } while(0)

//assign Neural Network parameters
typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    float *d_fc1_output, *d_fc2_output, *d_grad_hidden, *d_grad_output;
    float *d_input_batch;
    int *d_labels;
    float *d_loss;

    cublasHandle_t cublas_handle;

} NeuralNetwork;

typedef struct {
    double data_loading;
    double fwd_pass;
    double cross_entropy;
    double bwd_pass;
    double weight_updates;
    double total_time;
} TimingStats;

// cpu time measurement
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
    memset(bias, 0, size*sizeof(float));
}

//initializing the weight on the CPU DRAM and move to GPU
void initialize_random_weights(NeuralNetwork *nn){
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE*HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

//initializing the neural network parameters on the GPU VRAM
void initialize_neural_network(NeuralNetwork *nn){
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    //PERSISTENT BUFFERS
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nn->d_loss, BATCH_SIZE * sizeof(float)));

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    initialize_random_weights(nn);
}

__global__ void bias_sum(float *x, const float *bias, 
                       int batch_size, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx< batch_size *size){
        int bias_idx  = idx%size;
        x[idx]+=bias[bias_idx];
    }
}

__global__ void relu_forward_kernel(float *x, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        x[idx] = fmaxf(0.0f, x[idx]);
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
    if (i<batch_size * size){
        int bias_idx = i%size;
        atomicAdd(&grad_bias[bias_idx], grad[i]);
    }
}

__global__ void softmax_cross_entropy_backward_kernel(
    float *logits, 
    int *labels, 
    float *grad_output,
    float *loss_per_sample,
    int batch_size, 
    int num_classes
){
    int b = blockIdx.x;
    if (b>=batch_size) return;

    //using shared memory for this sample's logits
    extern __shared__ float shared[];
    float *sample_logits = shared;

    int tid = threadIdx.x;

    //load logits to shared memory
    if (tid<num_classes){
        sample_logits[tid]= logits[b*num_classes + tid];    
    }
    __syncthreads();

    //Find max for numerical stability
    __shared__ float max_logit;
    __shared__ float sum_exp;

    if (tid ==0){
        max_logit = sample_logits[0];
        for (int i =1; i<num_classes; ++i){
            if (sample_logits[i]>max_logit) max_logit = sample_logits[i];
        }
    }
    __syncthreads();

    //computing exp(logit - max)
    if (tid<num_classes){
        sample_logits[tid] = expf(sample_logits[tid]-max_logit);
    }
    __syncthreads();

    //compute sum of exponentials
    if (tid==0){
        sum_exp = 0.0f;
        for (int i = 0; i<num_classes; i++){
            sum_exp += sample_logits[i];
        }
    }
    __syncthreads();

    //cumpute softmax, gradient, and loss
    if (tid<num_classes){
        float prob = sample_logits[tid]/sum_exp;
        int label = labels[b];

        //Gradient: (prob - one_hot)/batch_size
        float grad = prob;
        if (tid== label){
            grad -= 1.0f;
        }

        grad/=(float)batch_size;
        grad_output[b*num_classes +tid]=grad;

        //loss contribution (only for correct class)
        if (tid == label){
            loss_per_sample[b] = -logf(fmaxf(prob, 1e-7f));
        }
    }
}

// seting up the training loop
void forward_pass(NeuralNetwork *nn, 
    int batch_size){
    const float alpha = 1.0f, beta = 0.0f;

    //Forward matmul 1: input *weights1
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                HIDDEN_SIZE, batch_size, INPUT_SIZE,
                &alpha, nn->d_weights1, HIDDEN_SIZE,
                nn->d_input_batch, INPUT_SIZE, &beta,
                nn->d_fc1_output, HIDDEN_SIZE));

    //Forward bias add 1
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255)/256; //ceildiv
    bias_sum<<<grid_hidden, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE);

    //Forward ReLU
    relu_forward_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, total_hidden);

    //Forward matmul2: hidden*weights2
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N,
                CUBLAS_OP_N, OUTPUT_SIZE, 
                batch_size, HIDDEN_SIZE,
                &alpha, nn->d_weights2, OUTPUT_SIZE,
                nn->d_fc1_output, HIDDEN_SIZE,
                &beta, nn->d_fc2_output, OUTPUT_SIZE));

    //Forward bias add 2 (no sync needed - loss computed on GPU)
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out +255)/256;
    bias_sum<<<grid_out, 256>>>(nn->d_fc2_output, nn->d_bias2,
            batch_size, OUTPUT_SIZE);
}

//backward propagation
void backward_pass(NeuralNetwork *nn, 
            int batch_size){
    const float alpha = 1.0f, beta=0.0f;

    //Zero gradients (async)
    CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, OUTPUT_SIZE * sizeof(float)));

    //Backward matmul 2a: weights2 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N,
                CUBLAS_OP_T, OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                &alpha, nn->d_grad_output, OUTPUT_SIZE,
                nn->d_fc1_output, HIDDEN_SIZE, &beta,
                nn->d_grad_weights2, OUTPUT_SIZE));

    //Backward bias2 gradients
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255)/256;
    bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_bias2, nn->d_grad_output,
    batch_size, OUTPUT_SIZE);

    //Backward matmul 2b: hidden gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T,
            CUBLAS_OP_N, HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
            &alpha, nn->d_weights2, OUTPUT_SIZE, nn->d_grad_output,
            OUTPUT_SIZE, &beta, nn->d_grad_hidden, HIDDEN_SIZE));

    //Backward ReLU
    int total_hidden= batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden +255)/256;
    relu_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_fc1_output, total_hidden);

    //Backward matmul 1a: weights1 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            HIDDEN_SIZE, INPUT_SIZE, batch_size, &alpha,
            nn->d_grad_hidden, HIDDEN_SIZE, nn->d_input_batch,
            INPUT_SIZE, &beta, nn->d_grad_weights1, HIDDEN_SIZE));

    //Backward bias1 gradients
    bias_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_bias1, nn->d_grad_hidden, batch_size, HIDDEN_SIZE);
}

// weight updates
void update_weights(NeuralNetwork *nn, float lr){
    float neg_lr = -lr;
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE*HIDDEN_SIZE,
            &neg_lr, nn->d_grad_weights1,1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE*OUTPUT_SIZE,
            &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2,1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
            &neg_lr, nn->d_grad_bias1,1, nn->d_bias1,1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
            &neg_lr, nn->d_grad_bias2,1, nn->d_bias2,1));

    //synchronize at the end of a iteration
    CUDA_CHECK(cudaDeviceSynchronize());
}

float compute_loss_on_gput(NeuralNetwork *nn, int batch_size){
    int shared_mem = OUTPUT_SIZE * sizeof(float);
    softmax_cross_entropy_backward_kernel<<<batch_size, 32, shared_mem>>>(
        nn->d_fc2_output, nn->d_labels, nn->d_grad_output,
        nn->d_loss, batch_size, OUTPUT_SIZE);
    
    float h_loss[BATCH_SIZE];
    CUDA_CHECK(cudaMemcpy(h_loss, nn->d_loss, batch_size*sizeof(float), cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i=0; i<batch_size; ++i){
        total_loss+= h_loss[i];
    }
    return total_loss/batch_size;
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
    
    int num_batches = TRAIN_SIZE/BATCH_SIZE;

    TimingStats stats = {0}; //initialize a struct

    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++){
            int start_idx = batch * BATCH_SIZE;

            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
           // int *batch_labels_cpu = &y_train[start_idx];
            
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(nn->d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(nn->d_labels, batch_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading+= get_time_diff(step_start, step_end); //create the get_time_diff function
            
            //forward propagation
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            forward_pass(nn, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.fwd_pass+= get_time_diff(step_start, step_end);
            
            //calculating loss
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float batch_loss = compute_loss_on_gput(nn, BATCH_SIZE);
            total_loss += batch_loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy+= get_time_diff(step_start, step_end);
           
            //backward propagation
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            backward_pass(nn, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.bwd_pass+= get_time_diff(step_start, step_end);
            
            //Optimizer weight update
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            update_weights(nn, LEARNING_RATE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.weight_updates +=get_time_diff(step_start, step_end);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss/num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    printf("\n === Cuda GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("Detailed Breakdown:\n");
    printf(" Data loading:  %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading/stats.total_time);
    printf(" Forward_pass: %6.3fs (%5.1f%%)\n", stats.fwd_pass, 100.0 * stats.fwd_pass / stats.total_time);
    printf(" Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy/stats.total_time);
    printf(" Backward pass: %6.3fs (%5.1f%%)\n", stats.bwd_pass, 100.0 * stats.bwd_pass / stats.total_time);
    printf(" Weight updates: %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates/stats.total_time);

}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
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
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden));
    CUDA_CHECK(cudaFree(nn->d_grad_output));   // add this
    CUDA_CHECK(cudaFree(nn->d_fc1_output));  
    CUDA_CHECK(cudaFree(nn->d_fc2_output));

    //free persistent buffers
    CUDA_CHECK(cudaFree(nn->d_input_batch));
    CUDA_CHECK(cudaFree(nn->d_labels));
    CUDA_CHECK(cudaFree(nn->d_loss));

    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

//  main
int main(void) {
    srand(42);  // fixed seed for reproducibility

    // Allocate data buffers
    float *X_train =(float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train =(int*)malloc(TRAIN_SIZE * sizeof(int));
    
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
   // initialize_random_weights(&nn);

    printf("Network: %d → %d → %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d  |  Epochs: %d  |  LR: %.4f\n\n",
           BATCH_SIZE, EPOCHS, LEARNING_RATE);

    // Train
    train_timed(&nn, X_train, y_train);

    // Cleanup
    free_network(&nn);
    free(X_train);
    free(y_train);

    return 0;
}