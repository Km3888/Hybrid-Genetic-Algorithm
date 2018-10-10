import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_Continuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net_Continuous, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim)-2)
    def forward(self, x):
        out = F.relu(self.input(x))
        mu = F.tanh(self.mu(out))
        action_log_std = self.action_log_std.expand_as(mu)
        return mu, action_log_std

class Net_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net_Baseline, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,1)
    def forward(self, x):
        out = F.relu(self.input(x))
        out = F.tanh(self.output(out))
        return out
