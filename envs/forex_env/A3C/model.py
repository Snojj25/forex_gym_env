import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class ActorCritic(nn.Module):

    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        # self.lstm1 = nn.LSTMCell(input_dim, 64)
        # self.lstm2 = nn.LSTMCell(64, 64)
        # self.lstm3 = nn.LSTMCell(64, 64)
        self.lstm = nn.LSTM(
            input_size = 9,
            hidden_size = 64,
            num_layers = 2,
            dropout = 0.25,
            batch_first = True
        )
        # self.fc1 = nn.Linear(2880, 1032)
        # self.fc2 = nn.Linear(1032, 64)
        # Fully connected
        self.critic_fc1 = nn.Linear(64, 1) 
        self.actor_fc1 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p = 0.25)
        # Weight initialization:
        self.apply(weights_init)
        self.actor_fc1.weight.data = torch.nn.init.normal_(self.actor_fc1.weight.data, mean=0, std=0.1)
        self.actor_fc1.bias.data.fill_(0.1)
        self.critic_fc1.weight.data = torch.nn.init.normal_(self.critic_fc1.weight.data, mean=0, std=1.0)
        self.critic_fc1.bias.data.fill_(0.1)
        # self.fc1.weight.data = torch.nn.init.normal_(self.fc1.weight.data, mean=0, std=0.5)
        # self.fc1.bias.data.fill_(0.1)
        # self.fc2.weight.data = torch.nn.init.normal_(self.fc2.weight.data, mean=0, std=0.5)
        # self.fc2.bias.data.fill_(0.1)
        self.train()

    def forward(self, inputs):
        # inputs, [(h_t,c_t), (h_t2,c_t2), (h_t3,c_t3)] = inputs
        # h_t, c_t = self.lstm1(inputs, (h_t, c_t))
        # h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
        # h_t3, c_t3 = self.lstm2(h_t2, (h_t3,c_t3))
        # hidden_list = [(h_t,c_t), (h_t2,c_t2), (h_t3,c_t3)]
        inputs, (hx,cx) = inputs
        x, (hx,cx)= self.lstm(inputs, (hx,cx))

        # x = self.dropout(self.fc1(h_t3))
        # x = self.dropout(self.fc2(x))
        x = self.dropout(x.squeeze()[-1])
        Vs = self.critic_fc1(x)     #.squeeze()
        Qs = self.actor_fc1(x)      #.squeeze()
        return Vs, Qs, (hx,cx)

