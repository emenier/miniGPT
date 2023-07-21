import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,d_in,dropout_freq=0.,factor=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in,factor*d_in),
            nn.ReLU(),
            nn.Linear(factor*d_in,d_in),
            nn.Dropout(dropout_freq)
        )
    
    def forward(self,x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self,d_in):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_in))
        self.beta = nn.Parameter(torch.zeros(d_in))

    def forward(self,x):
        #return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)
        centered = x - x.mean(-1,keepdim=True)
        scaled = centered/centered.std(-1,keepdim=True)

        return self.gamma * scaled + self.beta
