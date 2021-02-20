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
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = 64,
            num_layers = 3,
            batch_first = True,
            dropout = 0.5
        )
        num_outputs = len(action_space)
        self.critic_fc1 = nn.Linear(64, 64) 
        self.critic_fc2 = nn.Linear(64, 32)
        self.critic_fc3 = nn.Linear(32, 1) #Output = V(S)
        self.actor_fc1 = nn.Linear(64, 64)
        self.actor_fc2  = nn.Linear(64, 32)
        self.actor_fc3 = nn.Linear(32, num_outputs) #Output = Q(S,A)
        self.dropout = nn.Dropout(p = 0.25)
        # Weight initialization:
        self.apply(weights_init)
        self.actor_fc1.weight.data = torch.nn.init.normal_(self.actor_fc1.weight.data, mean=0, std=0.1)
        self.actor_fc1.bias.data.fill_(0.1)
        self.critic_fc1.weight.data = torch.nn.init.normal_(self.critic_fc1.weight.data, mean=0, std=1.0)
        self.critic_fc1.bias.data.fill_(0.1)
        self.actor_fc2.weight.data = torch.nn.init.normal_(self.actor_fc2.weight.data, mean=0, std=0.1)
        self.actor_fc2.bias.data.fill_(0.1)
        self.critic_fc2.weight.data = torch.nn.init.normal_(self.critic_fc2.weight.data, mean=0, std=1.0)
        self.critic_fc2.bias.data.fill_(0.1)
        self.actor_fc3.weight.data = torch.nn.init.normal_(self.actor_fc3.weight.data, mean=0, std=0.1)
        self.actor_fc3.bias.data.fill_(0.1)
        self.critic_fc3.weight.data = torch.nn.init.normal_(self.critic_fc3.weight.data, mean=0, std=1.0)
        self.critic_fc3.bias.data.fill_(0.1)
        self.train()

    def forward(self, inputs):
        out, (h_n, c_n) = self.lstm(inputs)
        x = self.critic_fc1(out)
        x = self.dropout(x)
        x = self.critic_fc2(x)
        x = self.dropout(x)
        x = self.critic_fc3(x)
        Vs = self.dropout(x)

        y = self.actor_fc1(out)
        y = self.dropout(y)
        y = self.actor_fc2(y)
        y = self.dropout(y)
        y = self.actor_fc3(y)
        Qs = self.dropout(y)

        return Vs, Qs, (h_n,c_n)