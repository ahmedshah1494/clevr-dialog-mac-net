import torch
from torch import nn
import numpy as np

def output_mask(x, xlens):
    mask = torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)
    for i,l in enumerate(xlens):
        mask[i,l:] = -float('inf')            
    return mask    

class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def get_alignment(self, x, y, xlens=None):
        """
            inputs
            x : sequence to be attended to (sequence of keys) having shape BxLxD
            y : The attention query having shape BxD
            
            returns
            A : the alignment matrix containing alignment weights
        """
        
        y = y.unsqueeze(2)
        E = torch.bmm(x, y)
        
        if xlens is not None:
            mask = output_mask(x, xlens)
            masked_E = E * mask
            # print(masked_E)
        else:
            masked_E = E
        
        A = nn.functional.softmax(masked_E, 1)        
        return A
    
    def forward(self, x, y, xlens=None):
        A = self.get_alignment(x, y, xlens)
        attended_x = torch.bmm(A.transpose(1,2), x)
        return attended_x

class GeneralAttention(DotAttention):
    def __init__(self, enc_dim, dec_dim):
        super(GeneralAttention, self).__init__()
        self.kproj = nn.Linear(enc_dim, enc_dim)
        self.qproj = nn.Linear(dec_dim, enc_dim)
        self.vproj = nn.Linear(enc_dim, enc_dim)
    
    def forward(self, x, y, xlens=None):
        k = self.kproj(x)
        q = self.qproj(y)
        v = self.vproj(x)
        
        A = self.get_alignment(k, y, xlens)
        attended_x = torch.bmm(v.transpose(0,1), A)
        return attended_x

class ConcatAttention(DotAttention):
    def __init__(self, enc_dim, dec_dim):
        super(ConcatAttention, self).__init__()
        self.W_a = nn.Linear(enc_dim+dec_dim, enc_dim)
        self.v = nn.Linear(enc_dim, 1)

    def get_alignment(self, x, y, xlens):
        tiled_y = torch.repeat_interleave(y.unsqueeze(1), x.shape[1], 1)
        E = self.v(self.W_a(torch.cat((x, tiled_y), 2)))

        mask = output_mask(x, xlens)
        masked_E = E + mask        
        
        A = nn.functional.softmax(masked_E, 1)        
        return A

    def forward(self, x, y, xlens=None):
        return super().forward(x, y, xlens=xlens)
    
        