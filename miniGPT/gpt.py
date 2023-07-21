import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from . import attention
from. import misc

class DecoderGPTBlock(nn.Module):

    def __init__(self, C, n_heads, max_block_size, dropout_freq=0.):

        super().__init__()

        self.self_attention = attention.CausalSelfAttention(
                        C, n_heads,  
                        max_block_size, dropout_freq=dropout_freq)

        self.attention_norm = misc.LayerNorm(C)

        self.ffn = misc.FFN(C,dropout_freq=dropout_freq)

        self.ffn_norm = misc.LayerNorm(C)

    def forward(self,x):
        # Dimensions stay the same at this level (B, T, C)
        normalised_attention_output = self.self_attention(
                                            self.attention_norm(x))

        x = x + normalised_attention_output

        normalised_fedforward = self.ffn(self.ffn_norm(x))

        return x + normalised_fedforward


class DecoderGPT(nn.Module):

    def __init__(self,vocab_size, C, n_layers, n_heads, 
                    max_block_size, dropout_freq=0.,
                    gpus_to_split=None):
        super().__init__()
        self.layers = nn.ModuleList([DecoderGPTBlock(
        #self.layers = nn.Sequential(*[DecoderGPTBlock(
            C, n_heads, max_block_size, dropout_freq=dropout_freq)
            for _ in range(n_layers)])

        self.x_embed = nn.Embedding(vocab_size,C)
        self.p_embed = nn.Embedding(max_block_size,C)
        self.dropout = nn.Dropout(dropout_freq)

        self.out_norm = misc.LayerNorm(C)
        self.out_layer = nn.Linear(C,vocab_size)

        self.x_embed.weight = self.out_layer.weight

        self.max_block_size = max_block_size
        
        self.gpus_to_split = gpus_to_split
        if gpus_to_split is not None:
            
            assert n_layers % (len(gpus_to_split)-1) == 0
            layer_per_gpu = int(n_layers/(len(gpus_to_split)-1))
            self.devs = []
            for i in range(len(gpus_to_split)-1):
                for j in range(layer_per_gpu):
                    dev = torch.device(f'cuda:{i+1:}')
                    self.layers[j+i*layer_per_gpu].to(dev)
                    self.devs.append(dev)



    def forward(self,x):
        in_dev = x.device
        
        if self.gpus_to_split is None:
            self.devs = [in_dev for _ in self.layers]

        B,T = x.shape

        x_embed = self.x_embed(x)
        p_embed = self.p_embed(torch.arange(T).to(x.device))
        embedding = self.dropout(x_embed + p_embed)
        for d,l in zip(self.devs,self.layers):
            embedding = l(embedding.to(d))
        #embedding = self.layers(embedding)
        out = self.out_norm(embedding.to(in_dev))

        return self.out_layer(out)

    def to(self, *args, **kwargs):
        
        self = super().to(*args, **kwargs) 
        if self.gpus_to_split is not None:
            for d,l in zip(self.devs,self.layers):
                l.to(d)
        
        return self

    def generate(self,x,generation_length):

        
        for _ in tqdm(range(generation_length)):
            
            logits = self(x[:,-self.max_block_size:])[:,-1]
            probabilities = F.softmax(logits,dim=-1)
            cur_tokens = torch.multinomial(probabilities,num_samples=1)
            x = torch.cat([x,cur_tokens],dim=-1)
        return x


