'''
多头注意力的实现
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, d_model:int, n_heads:int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 保证维度整体可除
        assert self.head_dim * n_heads == d_model, "d_model 必须能被 n_heads 整除"

        # 初始化映射矩阵
        self.q_proj = nn.Linear(d_model,d_model,bias=False)
        self.k_proj = nn.Linear(d_model,d_model,bias=False)
        self.v_proj = nn.Linear(d_model,d_model,bias=False)
        self.o_proj = nn.Linear(d_model,d_model,bias=False)

    def forward(self, x, mask=None):

        batch_size, seq_len, _ = x.shape

        # Q/K/V 线性映射 + 多头拆分、维度转置
        # [batch_size,seq_len,d_model] -> [batch_size,seq_len,n_heads,head_dim] -> [batch_size,n_heads,seq_len,head_dim]
        q = self.q_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)

        # 计算注意力分数
        # [batch_size,n_heads,seq_len,head_dim] * [batch_size,n_heads,head_dim,seq_len] -> [batch_size,n_heads,seq_len,seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2,-1))

        # 判断是否使用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0,-torch.inf)

        # 使用激活函数
        attn_probs = F.softmax(attn_scores,dim=-1)

        # 计算注意力权重
        # [batch_size,n_heads,seq_len,seq_len] * [batch_size,n_heads,seq_len,head_dim]- > [batch_size,n_head,seq_len,head_dim]
        attn_weight = torch.matmul(attn_probs,v)

        # 多头合并
        # [batch_size,n_head,seq_len,head_dim] -> [batch_size,seq_len,d_model]
        output = attn_weight.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)

        # 输出最终结果
        return self.o_proj(output)