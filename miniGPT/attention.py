import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttentionHead(nn.Module):

    def __init__(self,C, d_head, max_block_size, dropout_freq=0.):
        super().__init__()
        self.Q = nn.Linear(C, d_head, bias=False)
        self.K = nn.Linear(C, d_head, bias=False)
        self.V = nn.Linear(C, d_head, bias=False)
        
        self.register_buffer('tril', 
        torch.tril(torch.ones(max_block_size, max_block_size)))
        self.dropout = nn.Dropout(dropout_freq)

    def forward(self,inp):
        B, T, C = inp.shape
        queries = self.Q(inp) # B, T, D_head
        keys = self.K(inp) # B, T, D_head
        weight = queries @ keys.transpose(-2,-1) * keys.shape[-1]**-0.5 # B, T, T

        # Causal Masking
        masked_weight = weight.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        softmaxed_weight = F.softmax(masked_weight,dim=-1)
        attention = self.dropout(softmaxed_weight) # B, T, T

        # Value aggregation
        values = self.V(inp) # B, T, D_head
        return attention @ values # B, T, D_head

class MultiHeadAttention(nn.Module):

    def __init__(self, C, n_heads, max_block_size, 
                    dropout_freq=0.,attention_class=CausalAttentionHead):
        super().__init__()
        d_head = C//n_heads
        self.heads = nn.ModuleList(
            [attention_class(C, d_head, max_block_size, dropout_freq=0.)
                        for _ in range(n_heads)])
        self.proj = nn.Linear(d_head*n_heads,C)
        self.dropout = nn.Dropout(dropout_freq)


    def forward(self,x):

        concat = torch.cat([h(x) for h in self.heads],dim=-1) # B,T,D_out
        return self.proj(self.dropout(concat)) # B,T,C



