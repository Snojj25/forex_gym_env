import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


dataset = pd.read_csv(
        "C:/JURE_SNOJ/forex_gym/forex_gym_env/dataset/crypto/BTCUSD_M5.csv",
        names = ["date", "open", "high", "low", "close", "volume"]
        )

training_set = dataset.iloc[:180000, 1:].values
test_set = dataset.iloc[180000-60:, 1:].values

x_train = []
y_train = []
for i in range(60, 180000):
    x_train.append(training_set[i-60:i, :])
    y_train.append(training_set[i, :])
 
x_test = []
y_test = []
for i in range(60, 20000):
    x_test.append(training_set[i-60:i, :])
    y_test.append(training_set[i, :])
    
    
class LSTMRegressor(nn.Module):
    
     def __init__(self):
        super(LSTMRegressor, self).__init__()
        self.lstm1 = nn.LSTMCell(5, 64)
        self.lstm2 = nn.LSTMCell(64, 64)
        self.lstm3 = nn.LSTMCell(64, 64)
        self.fc1 = nn.Linear(64, 5)
        
     def forward(self, inputs):
        inputs, [(h_t,c_t), (h_t2,c_t2), (h_t3,c_t3)] = inputs
        h_t, c_t = self.lstm1(inputs, (h_t,c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
        h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
        hidden_list = [(h_t,c_t), (h_t2,c_t2), (h_t3,c_t3)]
        return self.fc1(h_t3[-1]), hidden_list
        
def save_checkpoint(state, filename = "LSTM_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state,filename)


def load_checkpoint(model, path ="LSTM_checkpoint.pth.tar"):
    print("=> Loading checkpoint")  
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        pass

#print(torch.cuda.is_available())
        
batch_size = 100
#num_batches = int(len(training_set/batch_size))
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

train_set = TensorDataset(x_train, y_train)
train_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False) 
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

model = LSTMRegressor()
loss = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = 0.01)



def train(model, optim, loss_fn, epochs, load_model = True):
    count = 0
    if load_model:
        load_checkpoint(model)
    for epoch in range(epochs):
        for x, y in train_loader:
            if count % 500 == 0:
                checkpoint = {
                    "state_dict" : model.state_dict(),
                    "optimizer" : optimizer.state_dict()
                }
                save_checkpoint(checkpoint)
            
            if count == 0:
                h_t  = torch.rand((60, 64), requires_grad = True)
                c_t  = torch.rand((60, 64), requires_grad = True)
                h_t2 = torch.rand((60, 64), requires_grad = True)
                c_t2 = torch.rand((60, 64), requires_grad = True)
                h_t3 = torch.rand((60, 64), requires_grad = True)
                c_t3 = torch.rand((60, 64), requires_grad = True)
            
                hidden_list = [(h_t, c_t), (h_t2, c_t2), (h_t3, c_t3)]

            
            inputs = x, hidden_list
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            out, hidden_list = model(inputs)
            # Calculate loss
            loss = loss_fn(out, y)
            # Calculate gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
            count += 1
            if count % 250 == 0:
                total = 0
                error = 0
                for x, y in test_loader:
                    
                    out, _ = model(x, hidden_list)
                    total += y.size(0)
                    error += (y - out)
                    
            accuracy = 100 - error/total 
            
            losses.append(loss.data)
            accuracies.append(accuracy)
            if count % 5 == 0:
                print("Iteration: {}, Loss: {}, Accuracy: {}"
                      .format(count, loss.data[0], accuracy))
            
        
        
        
        
model = LSTMRegressor()
optimizer = Adam(model.parameters(), lr = 0.001)   
loss = nn.MSELoss()
        
losses = []
accuracies = []  


if __name__ == "__main__":
    train(model, optimizer, loss, 5)
        
        
        
        
        
        
        
        
        
        
        
        