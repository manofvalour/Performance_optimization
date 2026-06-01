#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000

typedef struct {
    float *weights1, *weights2, *bias1, *bias2;
    float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2;
    float *grad_input1, *grad_input2;
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

void initialize_neural_network(NeuralNetwork *nn){
    nn->weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE *sizeof(float));
    nn->grad_weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_input2 = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_input1 = malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(nn->weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn-> bias2, OUTPUT_SIZE);
}

// forward pass
void matmul_a_b(float *A, float *B, float *C, 
                int m, int n, int l){

    for (int i=0; i<m; i++){
        for (int j = 0; j<n; j++){
            C[i*n+j]=0.0f;
            for (int k = 0; k<l; k++){
                C[i*n+j]+= A[i*l+k]*B[k*n+j];
            }
        }
    }
}

void bias_forward(float* x, float *bias, int batch_size, int size){
    for (int b = 0; b<batch_size; b++){
        for(int i =0; i< size; i++){
            x[b*size+i]+= bias[i];
        }
    }
}

void relu_forward(float *x, int size){
    for (int i = 0; i < size; i++){
        x[i] =fmaxf(0.0f, x[i]);
    }
}

// backward pass
void matmul_at_b(float *A, float *B, float *C, 
                        int m, int n, int l){
    for (int i= 0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i*n+j]=0.0f;
            for (int k=0; k<l;k++){
                C[i*n+j]+= A[k*m+i]*B[k*n+j]; //A_transpose @ B
            }
        }
    }
}

void matmul_a_bt(float *A, float *B, float *C,
                int m, int n, int l){
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i*n+j] =0.0f;
            for (int k=0; k<l; k++){
                C[i*n+j] += A[i*l+k]* B[j*l+k]; // A @ B_transpose
            }
        }
    }
}

void relu_backward(float *grad, float *x, int size){
    for (int i = 0; i<size; i++){
        grad[i]*= (x[i]>0);
    }
}

void bias_backward(float *grad, float *grad_bias, 
    int batch_size, int size){

    for (int i = 0; i<size; i++){
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++){
            grad_bias[i]+= grad[b*size+i];
        }
    }
}

// Zero out gradients
void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

// implementing the forward pass (wx+b)
void forward_timed(NeuralNetwork *nn, float *batch_input, 
                float *hidden, float *output, 
                int batch_size, int hidden_size, TimingStats *stats){

    struct timespec step_start, step_end;
    
    //first pass relu(w1x+b1)
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_a_b(batch_input, nn->weights1, hidden,
                batch_size, hidden_size, INPUT_SIZE); //wx
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_matmul1+= get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_forward(hidden, nn->bias1, batch_size, hidden_size); //+b
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_bias1+= get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    relu_forward(hidden, batch_size * hidden_size);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_relu+= get_time_diff(step_start, step_end);

    //second pass (w2x+b2)
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_a_b(hidden, nn->weights2, output,
                batch_size, OUTPUT_SIZE, hidden_size); //w2x
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_matmul2+= get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE); // +b2
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->fwd_bias2+= get_time_diff(step_start, step_end);
}

// implementing the backward pass(dLoss/dout)
void backward_timed(NeuralNetwork *nn, float *batch_input, float *hidden, 
                    float *output, int *batch_labels, int batch_size, 
                    TimingStats *stats){
    
    struct timespec step_start, step_end;

    zero_grad(nn->grad_weights1, INPUT_SIZE*HIDDEN_SIZE);
    zero_grad(nn->grad_weights2, HIDDEN_SIZE*OUTPUT_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    //loss backward linear layer2
    clock_gettime(CLOCK_MONOTONIC, &step_start); 
    matmul_at_b(hidden, output, nn->grad_weights2,
                batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_matmul2+=get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_a_bt(output, nn->weights2, nn->grad_input2,
                batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_output_grad+=get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_backward(output, nn->grad_bias2, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_bias2+=get_time_diff(step_start, step_end);

    //linear layer1
    clock_gettime(CLOCK_MONOTONIC, &step_start);
    relu_backward(nn->grad_input2, hidden, batch_size * HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_relu+=get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    matmul_at_b(batch_input, nn->grad_input2, nn->grad_weights1,
                INPUT_SIZE, HIDDEN_SIZE, batch_size);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_matmul1+=get_time_diff(step_start, step_end);

    clock_gettime(CLOCK_MONOTONIC, &step_start);
    bias_backward(nn->grad_input2, nn->grad_bias1, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->bwd_bias1+=get_time_diff(step_start, step_end);
}

void softmax_forward(float* output, int num_classes, int batch_size){
    for (int b = 0; b<batch_size; b++){
        float *row = &output[b*num_classes];

        float max = row[0];
        for (int c=1; c<num_classes; c++){
            if (row[c]>max) max = row[c];
        }

        float sum = 0.0f;
        for (int c=0; c<num_classes; c++){
            row[c] = expf(row[c]- max);
            sum += row[c];
        }

        for (int c=0; c<num_classes; c++){
            row[c]/=sum;
        }
    }
}

float cross_entropy_loss(float* output, int* labels, int batch_size){
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int true_class = labels[b];
        float p = output[b * OUTPUT_SIZE + true_class];

        // Clamp to [1e-7, 1.0] so log never hits -inf
        if (p < 1e-7f) p = 1e-7f;

        total_loss += -logf(p);
    }

    return total_loss / (float)batch_size;
}

void softmax_backward(float *output, int *labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        int true_class = labels[b];

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            // Subtract 1 from the true class probability
            if (i == true_class) {
                output[b * OUTPUT_SIZE + i] -= 1.0f;
            }
            // Divide by batch size (gradient of the mean)
            output[b * OUTPUT_SIZE + i] /= (float)batch_size;
        }
    }
}


void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &step_start);

    // weights1: (INPUT_SIZE x HIDDEN_SIZE)
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->weights1[i] -= LEARNING_RATE * nn->grad_weights1[i];
    }

    // bias1: (HIDDEN_SIZE)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }

    // weights2: (HIDDEN_SIZE x OUTPUT_SIZE)
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        nn->weights2[i] -= LEARNING_RATE * nn->grad_weights2[i];
    }

    // bias2: (OUTPUT_SIZE)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias2[i] -= LEARNING_RATE * nn->grad_bias2[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &step_end);
    stats->weight_updates += get_time_diff(step_start, step_end);
}


// implementing training loop
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train){
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE/BATCH_SIZE;

    TimingStats stats = {0}; //initialize a struct

    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++){
            int start_idx = batch * BATCH_SIZE;

            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading+= get_time_diff(step_start, step_end); //create the get_time_diff function
            
            //forward propagation
            forward_timed(nn, batch_input, hidden, 
                        output, BATCH_SIZE, HIDDEN_SIZE, 
                        &stats);
            
            //softmax
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            softmax_forward(output, OUTPUT_SIZE, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.fwd_softmax+=get_time_diff(step_start, step_end);

            //calculating loss
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(output, batch_labels,
                                            BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start,
                                                    step_end);

            //backward propagation
            //softmax_backward
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            softmax_backward(output, batch_labels, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.bwd_softmax+= get_time_diff(step_start, step_end);
            
            backward_timed(nn, batch_input, hidden, output, 
                        batch_labels, BATCH_SIZE, &stats);
            
            //Optimizer weight update
            update_weights_timed(nn,&stats);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss/num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    printf("\n === C CPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("Detailed Breakdown:\n");
    printf(" Data loading:  %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading/stats.total_time);
    
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf(" Forward_pass: %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);

    printf(" Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy/stats.total_time);
   
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1+stats.bwd_softmax;
    printf(" Backward pass: %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);

    printf(" Weight updates: %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates/stats.total_time);

    free(hidden);
    free(output);
}

//Loading dataset
int read_int_big_endian(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

// normalize data using MNIST mean and std
void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
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

// ============================================================
//  Accuracy helper — runs a forward pass, counts correct preds
// ============================================================
float compute_accuracy(NeuralNetwork *nn, float *X, int *y, int n_samples) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    int correct = 0;

    int num_batches = n_samples / BATCH_SIZE;
    for (int b = 0; b < num_batches; b++) {
        float *batch_x = &X[b * BATCH_SIZE * INPUT_SIZE];
        int   *batch_y = &y[b * BATCH_SIZE];

        matmul_a_b(batch_x, nn->weights1, hidden,
                   BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
        bias_forward(hidden, nn->bias1, BATCH_SIZE, HIDDEN_SIZE);
        relu_forward(hidden, BATCH_SIZE * HIDDEN_SIZE);
        matmul_a_b(hidden, nn->weights2, output,
                   BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
        bias_forward(output, nn->bias2, BATCH_SIZE, OUTPUT_SIZE);
        softmax_forward(output, OUTPUT_SIZE, BATCH_SIZE);

        for (int i = 0; i < BATCH_SIZE; i++) {
            // argmax over output row
            int pred = 0;
            float best = output[i * OUTPUT_SIZE];
            for (int c = 1; c < OUTPUT_SIZE; c++) {
                if (output[i * OUTPUT_SIZE + c] > best) {
                    best = output[i * OUTPUT_SIZE + c];
                    pred = c;
                }
            }
            if (pred == batch_y[i]) correct++;
        }
    }

    free(hidden);
    free(output);
    return (float)correct / (float)(num_batches * BATCH_SIZE) * 100.0f;
}

// ============================================================
//  Free all network memory
// ============================================================
void free_neural_network(NeuralNetwork *nn) {
    free(nn->weights1);   free(nn->weights2);
    free(nn->bias1);      free(nn->bias2);
    free(nn->grad_weights1); free(nn->grad_weights2);
    free(nn->grad_bias1); free(nn->grad_bias2);
    free(nn->grad_input1); free(nn->grad_input2);
}


//  main
int main(void) {
    srand(42);  // fixed seed for reproducibility

    // Allocate data buffers
    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int   *y_train = malloc(TRAIN_SIZE * sizeof(int));
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

    printf("Network: %d → %d → %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d  |  Epochs: %d  |  LR: %.4f\n\n",
           BATCH_SIZE, EPOCHS, LEARNING_RATE);

    // Train
    train_timed(&nn, X_train, y_train);

    // Final accuracy
    float acc = compute_accuracy(&nn, X_train, y_train, TRAIN_SIZE);
    printf("\nTrain accuracy: %.2f%%\n", acc);

    // Cleanup
    free_neural_network(&nn);
    free(X_train);
    free(y_train);

    return 0;
}