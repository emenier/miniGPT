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
    """ Not used, inefficient"""
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



class CausalSelfAttention(nn.Module):
    """
    Code pulled from : https://github.com/karpathy/nanoGPT/blob/master/model.py 
    for the flash attention.
    The multi heads are also computed as one "truncated" attention.
    """
    def __init__(self, C, n_heads, max_block_size, 
                    dropout_freq=0.):
        super().__init__()
        assert C % n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        # output projection
        self.c_proj = nn.Linear(C, C, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout_freq)
        self.resid_dropout = nn.Dropout(dropout_freq)
        self.n_head = n_heads
        self.n_embd = C
        self.dropout = dropout_freq
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(max_block_size, max_block_size))
                                        .view(1, 1, max_block_size, max_block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: 
        #  (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                attn_mask=None, dropout_p=self.dropout if self.training else 0, 
                is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head 
        #                                                  outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
