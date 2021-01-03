
from mido import MidiFile
from torch.utils import data
from torch.distributions import Bernoulli
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os#, load
import sys, pickle
import numpy as np

os.chdir("C:/Users/emmis/OneDrive/Skrivebord/02456-Deep Learning/Projekt/maestro-v2.0.0")

with open('split_data.pickle', 'rb') as f:
    tn, ts, val = pickle.load(f)


class MyRecurrentNet(nn.Module):
    def __init__(self, vocab_size):
        super(MyRecurrentNet, self).__init__()
        
        # Recurrent layer
        self.lstm = nn.LSTM(input_size=vocab_size,
                         hidden_size=200,
                         num_layers=1,
                         bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=200,
                            out_features=vocab_size,
                            bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        # x = nn.Tanh(x)
        
        return x

class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        
        return X, y
    
   
def make_batches(inputs, targets, batchsize):
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    inputs = inputs.reshape(inputs.shape[0], batchsize, inputs.shape[1], inputs.shape[2])
    targets = targets.reshape(targets.shape[0], batchsize, targets.shape[1], targets.shape[2])
    
    return inputs, targets

#%% Make batches

inputs_train = []
targets_train = []
inputs_test = []
targets_test = []
inputs_val = []
targets_val = []

for train in tn:
    inputs_train.append(train[:,:-1])
    targets_train.append(train[:,1:])

for test in ts:
    inputs_test.append(test[:,:-1])
    targets_test.append(test[:,1:])

for vali in val:
    inputs_val.append(vali[:,:-1])
    targets_val.append(vali[:,1:])
    

inputs_val, targets_val = make_batches(inputs_val, targets_val, batchsize=1)
inputs_train, targets_train = make_batches(inputs_train, targets_train, batchsize=1)
inputs_test, targets_test = make_batches(inputs_test, targets_test, batchsize=1)

validation_set = Dataset(inputs_val, targets_val)
training_set = Dataset(inputs_train, targets_train)
test_set = Dataset(inputs_test, targets_test)
    
#%% Train network

num_epochs = 200
net = MyRecurrentNet(inputs_val.shape[2])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)

training_loss, validation_loss = [], []

for i in range(num_epochs):
    
    epoch_training_loss = 0
    epoch_validation_loss = 0
    
    net.eval()
    
    for inputs, targets in validation_set:
        
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        inputs = inputs.permute(0,2,1)
        
        targets = torch.from_numpy(targets).type(torch.FloatTensor)
        
        outputs = net.forward(inputs)
        outputs = outputs.view((1, outputs.shape[1],outputs.shape[0]))
        
        dist = Bernoulli(logits=outputs)
        loss = -dist.log_prob(Bernoulli(targets))
        loss = loss.view(-1)[-1]
        
        epoch_validation_loss += loss.detach().numpy()
        
    net.train()
    
    for inputs, targets in training_set:
        
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        inputs = inputs.permute(0,2,1)
        
        targets = torch.from_numpy(targets).type(torch.FloatTensor)
        
        outputs = net.forward(inputs)
        outputs = outputs.view((1, outputs.shape[1],outputs.shape[0]))
        
        dist = Bernoulli(logits=outputs)
        loss = -dist.log_prob(Bernoulli(targets))
        loss = loss.view(-1)[-1]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_training_loss += loss.detach().numpy()
        
    training_loss.append(epoch_training_loss/len(training_set))
    validation_loss.append(epoch_validation_loss/len(validation_set))
        
    if i % 10 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

#%%

epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()
    
