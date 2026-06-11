'''
相较于Layer Norm，RMSNorm是可以减少计算，来加快计算速度
RMSNorm计算的是输入的均方根，而不是像Layer Norm那样计算均值和方差。
'''

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int,eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数α

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        return self.weight * self.norm(x.float()).type_as(x)