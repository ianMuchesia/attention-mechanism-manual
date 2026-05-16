import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        
        
        
    
    def compute_qkv(self,X):
        
        Q = self.W_q(X)
        
        print(f"Q shape is :{Q.shape}")
        K = self.W_k(X)
        
        
        print(f"K shape is :{K.shape}")
        V = self.W_v(X)
        
        print(f"V shape is :{V.shape}")
        
        return Q,K,V
        
        
    def compute_scores(self,Q,K):
        
        scores = Q @ K.transpose(-2,-1)
        
        scores = scores/math.sqrt(K.shape[-1])
        
        return scores
        
        