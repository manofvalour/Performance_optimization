import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(1)

TRAIN_SIZE = 10000;
epoches = 10;
learning_rates = 1e-2
batch_size = 8

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

    torch.set_float32_matmul_precision('high')
    #loading the data to memory
    #X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)
    #y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)

    #X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
    #y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)
    # Extract data and labels from torchvision datasets

    X_train_tensor = train_dataset.data.float() / 255.0
    y_train_tensor = train_dataset.targets.long()

    X_test_tensor = test_dataset.data.float() / 255.0
    y_test_tensor = test_dataset.targets.long()


    #normalizing the data
    mean, std = 0.1307, 0.3081;
    X_train_np = (X_train_tensor-mean)/std
    X_test_np = (X_test_tensor-mean)/std

    ## moving data to gpu vram
    #train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(-1,1,28,28)).to('cuda') #BDTC
    ##train_labels = torch.from_numpy(y_train_tensor[:TRAIN_SIZE]).long().to('cuda')
    #test_data = torch.from_numpy(X_test_np.reshape(-1,1,28,28)).to('cuda')
    #test_labels = torch.from_numpy(y_test_tensor).long().to('cuda')

    train_data = X_train_np[:TRAIN_SIZE].reshape(-1, 1,28,28).to('cuda')
    train_labels = y_train_tensor[:TRAIN_SIZE].to('cuda')
    test_data = X_test_np.reshape(-1,1,28,28).to('cuda')
    test_labels = y_test_tensor.to('cuda')

    return train_data, train_labels, test_data, test_labels


## Creating the model
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features,num_classes):
        super(MLP, self).__init__()
        self.fc1= nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes);

    def forward(self,x):
        x = x.reshape(batch_size, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MLP(in_features = 784, hidden_features=256, num_classes=10).to('cuda');

##implenting kaiming he wieght initialization method
with torch.no_grad():
    fan_in_fc1 = model.fc1.weight.size(1)
    scale_fc1 = (6.0/fan_in_fc1)**0.5
    model.fc1.weight.uniform_(-scale_fc1, scale_fc1)
    model.fc1.bias.zero_()

    fan_in_fc2 = model.fc2.weight.size(1)
    scale_fc2 = (6.0/fan_in_fc2)**0.5
    model.fc2.weight.uniform_(-scale_fc2,scale_fc2)
    model.fc2.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rates)

## Creating the Training loop function
def train_timed(model, criterion, optimizer, batch_size:int,
                timing_stats:dict, epoch_losses:list,
                train_data, train_labels):

    model.train()
    epoch_loss =0.0
    iters_per_epoch = math.ceil(TRAIN_SIZE/batch_size)

    for i in range(iters_per_epoch):

        ##profiling data loader
        data_start = time.time()
        data = train_data[i *batch_size:(i+1)*batch_size]
        target = train_labels[i*batch_size: (i+1)*batch_size]
        data_end = time.time()

        timing_stats['data_loading']+=data_end - data_start

        optimizer.zero_grad()
        
        #profiling model training forward pass
        forward_start = time.time()
        outputs = model(data)
        forward_end = time.time()
        timing_stats['forward']+=forward_end- forward_start

        ## profiling loss computation
        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation']+=loss_end-loss_start

        ## profiling backward pass
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start

        ## profiling update optimizer
        update_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        update_end=time.time()
        timing_stats['weight_updates'] += update_end - update_start

    epoch_losses.append(epoch_loss/iters_per_epoch)
    return epoch_loss/iters_per_epoch

## function for monitoring gradient health
def check_gradients(gradients, name):
    for param_name, grad in gradients.items():
        grad_norm = np.linalg.norm(grad)
        grad_max = np.max(np.abs(grad))
        print(f"{name} - {param_name}: norm = {grad_norm:.6f}, max={grad_max:.6f}")

        if grad_norm > 10:
            print(f"WARNING: Large gradient norm in {param_name}")
        if grad_max < 1e-6:
            print(f" WARNING: Very small gradients in {param_name}")

## implementing backward propagation of NN
def backward_batch(X, Z1, A1, Z2, y_true, W1, W2):
    batch_size = X.shape

    probs = softmax(Z2)
    grad_Z2 = probs.copy()
    grad_Z2[np.arange(batch_size), y_true]-=1
    grad_Z2/=batch_size

    grad_W2 = A1.T @ grad_Z2
    grad_b2 = np.sum(grad_Z2, axis = 0)

    grad_A1 = grad_Z2 @ W2.T
    grad_Z1 = grad_A1 * relu_derivative(Z1)

    grad_W1 = X.T @ grad_Z1
    grad_b1 = np.sum(grad_Z1,axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2

if __name__ == "__main__":
    num_epoch = 10
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0
    }
    epoch_losses = []
    train_data, train_labels, _,_ = download_data()
    for epoch in range(num_epoch):
        loss= train_timed(model, criterion, 
                    optimizer, batch_size, 
                    timing_stats, epoch_losses,
                    train_data,train_labels)
        print(f"Epoch {epoch}, loss: {loss}")
    print(timing_stats)
        
