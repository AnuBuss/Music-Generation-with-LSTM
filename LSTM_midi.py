# Build our Deep Learning Architecture

# from keras import layers
# from keras import models
# import keras
# from keras.models import Model
# import tensorflow as tf
# from keras.layers.advanced_activations import *


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

    
#%% Neural Net definition


class MyRecurrentNet(nn.Module):
    def __init__(self, vocab_size):
        super(MyRecurrentNet, self).__init__()
        
        # Recurrent layer
        self.lstm1 = nn.LSTM(input_size=vocab_size,
                         hidden_size=1024,
                         num_layers=1,
                         bidirectional=False)
        
        self.l1 = nn.Linear(in_features=1024,
                            out_features=1024,
                            bias=True)
        
        self.l2 = nn.Linear(in_features=1024,
                            out_features=512,
                            bias=True)
        
        self.l3 = nn.Linear(in_features=512,
                            out_features=512,
                            bias=True)
        
        
        self.lstm2 = nn.LSTM(input_size=512,
                         hidden_size=512,
                         num_layers=1,
                         bidirectional=False)
        
        self.l4 = nn.Linear(in_features=512,
                            out_features=256,
                            bias=True)
        
        self.l5 = nn.Linear(in_features=256,
                            out_features=256,
                            bias=True)
        
        
        self.lstm3 = nn.LSTM(input_size=256,
                         hidden_size=128,
                         num_layers=1,
                         bidirectional=False)
        
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
        self.batchnorm = nn.BatchNorm1d(7963)
        
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        # Output layer
        self.l_out = nn.Linear(in_features=128,
                            out_features=vocab_size,
                            bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm1(x)
        x = self.leaky_relu(x)
        x = self.batchnorm(x)
        x = self.dropout1(x)
        
        a = self.l1(x)
        a = self.tanh(a)
        a = self.softmax(a)
        
        multiplied = torch.mul(x, a)
        sent_representation = self.l2(multiplied)
        
        x = self.l3(sent_representation)
        x = self.leaky_relu(x)
        x = self.batchnorm(x)
        x = self.dropout2(x)
        
        x, (h, c) = self.lstm2(x)
        x = self.leaky_relu(x)
        x = self.batchnorm(x)
        x = self.dropout2(x)
        
        a = self.l3(x)
        a = self.tanh(a)
        a = self.softmax(a)
        
        multiplied = torch.mul(x, a)
        sent_representation = self.l4(multiplied)
        
        x = self.l5(sent_representation)
        x = self.leaky_relu(x)
        x = self.batchnorm(x)
        x = self.dropout2(x)
        
        x, (h, c) = self.lstm3(x)
        x = self.leaky_relu(x)
        x = self.batchnorm(x)
        x = self.dropout2(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm3.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        x = self.softmax(x)
        
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
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

training_loss, validation_loss = [], []
#%% Use Cuda
net.cuda()

#%% Begin Training
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
        
        loss = criterion(outputs, targets)
        
        epoch_validation_loss += loss.detach().numpy()
        
    net.train()
    
    for inputs, targets in training_set:
        
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        inputs = inputs.permute(0,2,1)
        
        targets = torch.from_numpy(targets).type(torch.FloatTensor)
        
        outputs = net.forward(inputs)
        outputs = outputs.view((1, outputs.shape[1],outputs.shape[0]))
        
        loss = criterion(outputs, targets)
        
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


with open('LSTM_data.pickle', 'wb') as f:
    pickle.dump([training_loss, validation_loss, ts], f)

#%%

# import random

# start_index = random.randint(0, len(ts[0])- 50 - 1)
    
# generated_midi = ts[0][start_index: start_index + 50]

# for temperature in [0.7, 2.7]:
#         print('------ temperature:', temperature)
#         generated_midi = ts[0][start_index: start_index + 50]
#         for i in range(680):
#             samples = generated_midi[i:]
#             expanded_samples = np.expand_dims(samples, axis=0)
#             preds = model.predict(expanded_samples, verbose=0)[0]
#             preds = np.asarray(preds).astype('float64')

#             next_array = sample(preds, temperature)
           
#             midi_list = []
#             midi_list.append(generated_midi)
#             midi_list.append(next_array)
#             generated_midi = np.vstack(midi_list)
            

#         generated_midi_final = np.transpose(generated_midi,(1,0))
#         output_notes = matrix_to_midi(generated_midi_final, random=1)
#         midi_stream = stream.Stream(output_notes)
#         midi_file_name = ('lstm_out_{}.mid'.format(temperature))
#         midi_stream.write('midi', fp=midi_file_name)
#         parsed = converter.parse(midi_file_name)
#         for part in parsed.parts:
#             part.insert(0, instrument.Piano())
#         parsed.write('midi', fp=midi_file_name)