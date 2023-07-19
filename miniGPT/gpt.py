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

        self.self_attention = attention.MultiHeadAttention(
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
                    max_block_size, dropout_freq=0.):
        super().__init__()
        self.layers = nn.Sequential(*[DecoderGPTBlock(
            C, n_heads, max_block_size, dropout_freq=dropout_freq)
            for _ in range(n_layers)])

        self.x_embed = nn.Embedding(vocab_size,C)
        self.p_embed = nn.Embedding(max_block_size,C)
        self.out_norm = misc.LayerNorm(C)
        self.out_layer = nn.Linear(C,vocab_size)
        self.max_block_size = max_block_size

    def forward(self,x):
        B,T = x.shape
        x_embed = self.x_embed(x)
        p_embed = self.p_embed(torch.arange(T).to(x.device))
        embedding = x_embed + p_embed
        out = self.out_norm(self.layers(embedding))

        return self.out_layer(out)



    def generate(self,x,generation_length):

        
        for _ in tqdm(range(generation_length)):
            
            logits = self(x[:,-self.max_block_size:])[:,-1]
            probabilities = F.softmax(logits,dim=-1)
            cur_tokens = torch.multinomial(probabilities,num_samples=1)
            x = torch.cat([x,cur_tokens],dim=-1)
        return x


