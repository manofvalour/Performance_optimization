import numpy as np
from torch.nn.modules import linear
import math
import torch
import time

TRAIN_SIZE = 10000
epoches = 10;
learning_rates = 1e-2
batch_size = 8

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def initialize_weights(input_size, output_size):
    scale = np.sqrt(6.0/input_size)
    return (np.random.rand(input_size, output_size)*2.0 -1.0)*scale;

def initialize_bias(output_size):
    return np.zeros((1,output_size))

def linear_forward(x, weights, bias):
    return x @ weights +bias

def linear_backward(grad_output, weights, x):

    grad_input = grad_output @ weights.T #8*10, 
    grad_weights = x.T @ grad_output
    grad_bias = np.sum(grad_output, axis=0)

    return grad_input, grad_weights, grad_bias


def softmax(x):
    exp_x = np.exp(x-np.max(x, axis=1, keepdims=True))
    den = np.sum(exp_x, axis = 1, keepdims=True)

    return exp_x/den

def cross_entropy_loss(y_pred, y_true):

    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    epsilon = 1e-10 # Add a small epsilon for numerical stability
    probabilities = np.clip(probabilities, epsilon, 1. - epsilon) # Clip probabilities to avoid log(0)
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    loss = -np.sum(correct_log_probs)/batch_size

    return loss


## creating a neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        fc1_input = np.reshape(x, (batch_size, -1))

        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache

        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output, self.weights2, relu_output)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1,grad_weights1, grad_bias1 = linear_backward(grad_relu,self.weights1, x)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2

    def update_weights(self, grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate):
        self.weights1 -= learning_rate * grad_weights1
        self.bias1 -= learning_rate * grad_bias1
        self.weights2-= learning_rate * grad_weights2
        self.bias2-= learning_rate * grad_bias2

def download_data():
    import torch, torchvision
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
    )

#    torch.set_float32_matmul_precision('high')

    X_train_tensor = train_dataset.data.float() / 255.0
    y_train_tensor = train_dataset.targets.long()

    X_test_tensor = test_dataset.data.float() / 255.0
    y_test_tensor = test_dataset.targets.long()


    #normalizing the data
    mean, std = 0.1307, 0.3081;
    X_train_np = (X_train_tensor.cpu().numpy()-mean)/std
    X_test_np = (X_test_tensor.cpu().numpy()-mean)/std


    train_data = X_train_np[:TRAIN_SIZE].reshape(-1, 1,28,28)
    train_labels = y_train_tensor[:TRAIN_SIZE].cpu().numpy()
    test_data = X_test_np.reshape(-1,1,28,28)
    test_labels = y_test_tensor.cpu().numpy()

    return train_data, train_labels, test_data, test_labels


def test(np_model, torch_model, batch_data, batch_data_np):

    state_dict = torch.load("initialized_weights.pth", map_location='cpu')
    np_model.weights1 = state_dict['fc1.weight'].T.numpy()
    np_model.bias1 = state_dict['fc1.bias'].numpy()

    np_model.weights2 = state_dict['fc2.weight'].T.numpy()
    np_model.bias2 = state_dict['fc2.bias'].numpy()

    batch_size = batch_data.shape[0]
    batch_data = torch.reshape(batch_data, (batch_size, -1))

    pytorch_hidden_output = torch_model.relu(torch_model.fc1(batch_data))
    pytorch_final_output = torch_model.fc2(pytorch_hidden_output)

    batch_data_np = np.reshape(batch_data_np.to('cpu'), (batch_size, -1))
    np_fc1_output = linear_forward(batch_data_np, np_model.weights1, np_model.bias1)
    np_relu_output = relu(np_fc1_output)
    np_fc2_output = linear_forward(np_relu_output, np_model.weights2, np_model.bias2)

    assert np.allclose(pytorch_hidden_output.detach().cpu().numpy(), np_relu_output,atol=1e-6)
    assert np.allclose(pytorch_final_output.detach().cpu().numpy(), np_fc2_output,atol=1e-6)
    print("True! pytorch and numpy is equal")

## Creating the Training loop function
def train_timed(model, batch_size:int,
                timing_stats:dict, epoch_losses:list,
                train_data, train_labels):

    epoch_loss =0.0
    iters_per_epoch = math.ceil(TRAIN_SIZE/batch_size)

    for i in range(iters_per_epoch):

        ##profiling data loader
        data_start = time.time()
        data = train_data[i *batch_size:(i+1)*batch_size]
        target = train_labels[i*batch_size: (i+1)*batch_size]
        data_end = time.time()

        timing_stats['data_loading']+=data_end - data_start

        #profiling model training forward pass
        forward_start = time.time()
        logits, cache = model.forward(data)
        forward_end = time.time()
        timing_stats['forward']+=forward_end- forward_start

        ## profiling loss computation
        loss_start = time.time()
        loss = cross_entropy_loss(logits, target)
        epoch_loss += loss
        loss_end = time.time()
        timing_stats['loss_computation']+=loss_end-loss_start

        ## profiling backward pass
        backward_start = time.time()
        # Calculate grad_output for the final layer (logits)
        probabilities = softmax(logits)
        batch_size_actual = logits.shape[0] # Use actual batch size for last batch
        grad_output_logits = probabilities
        one_hot_target = np.zeros_like(probabilities)
        one_hot_target[np.arange(batch_size_actual), target] = 1
        grad_output_logits -= one_hot_target
        grad_output_logits /= batch_size_actual # Average over batch

        grad_weight1, grad_bias1, grad_weight2, grad_bias2 = model.backward(grad_output_logits, cache)
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start

        ## profiling update optimizer
        update_start = time.time()
        model.update_weights(grad_weight1, grad_bias1, grad_weight2, grad_bias2, learning_rates)
        update_end=time.time()
        timing_stats['weight_updates'] += update_end - update_start

    epoch_losses.append(epoch_loss/iters_per_epoch)
    return epoch_loss/iters_per_epoch

if __name__== "__main__":
   # from neural_network_pytorch import MLP,download_data
    # initializing the models
    #pytorch_model = MLP(in_features = 784, hidden_features=256, num_classes=10).to('cuda')
    #np_model = NeuralNetwork(input_size = 784, hidden_size=256, output_size=10)
    #print('models initialized')

    #Save the initialized state_dict to a file
   # torch.save(pytorch_model.state_dict(), 'initialized_weights.pth')

    #importing the data
    #py_train_data, _, _,_ = download_data()
    #np_train_data, _, _, _ = download_data()
   # print('data downloaded successully')

    ## running checks
   # test(np_model, pytorch_model, py_train_data, np_train_data)

    num_epoch = 10
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0
    }
    epoch_losses = []

    train_data, train_label, _,_ = download_data()
    model = NeuralNetwork(input_size = 784, hidden_size=256, output_size=10)

    for epoch in range(num_epoch):
        loss= train_timed(model,
                    batch_size,
                    timing_stats, epoch_losses,
                    train_data,train_label)
        print(f"Epoch {epoch}, loss: {loss}")
    print(timing_stats)