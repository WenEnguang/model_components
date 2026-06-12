'''
GQA (Grouped Query Attention)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA(nn.Module):
    def __init__(self, d_model,n_heads,n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0, 'Model dimension must be divisible by number of heads'
        assert n_heads % n_kv_heads == 0,'Number of heads must be divisible by number of key-value heads'
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads    # 每一组头的数量

        # 初始化线性层，但是要注意K和V的头数
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model)

    def repeat_kv(self, x, n_rep):
        # GQA的关键函数，用于将K和V的头数重复到与Q相同的数量，
        # n_rep是指用同一套的K和V头来处理多个Q头，在数值上是等于Group_size的大小
        if n_rep ==  1:
            return x    # 不需要重复，直接返回原张量
        batch_size, n_kv_heads, seq_len, head_dim = x.shape

        '''
        关于使用repeat函数和expand函数的区别：
        - repeat函数会复制张量的元素，这在某些情况下可能会导致内存使用增加。真实复制数据，分配新内存。
        - expand函数则是在不复制实际数据的情况下，通过改变张量的形状来实现重复效果，更加高效，效果类似与指针/快照的效果，张量在内存上不连续（多个逻辑位置指向同一块内存）。
        关于reshape和view函数的区别：
        - reshape函数会返回一个新张量，但不保证返回的是原张量的视图。处理逻辑是，先检查是否连续，如不连续，就自动调用contiguous()方法复制一份，再执行view操作。
        - view函数则会返回原张量的视图，因此在内存使用上更加高效。但是前提是张量在内存中必须是连续存储的。

        所以，expand之后需要使用reshape函数来确保张量是连续的，否则在后续操作中可能会出现错误，或者手写 contiguous().view() 来强制连续。
        
        '''
        return (
            x[:, :, None, :, :] # 在头数后面加上一维维度
            .expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)   # 这里为什么不适用repeat函数？
            .reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim) # 这里为什么不再使用view函数？
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 1.投影
        # q矩阵尺寸变化
        # self.q_proj(x): [batch_size, seq_len, d_model] * [d_model, n_heads * head_dim] -> [batch_size, seq_len, n_heads * head_dim]
        # [batch_size, seq_len, n_heads * head_dim] -> [batch_size, seq_len, n_heads, head_dim] --transpose(1, 2)--> [batch_size, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # K/V矩阵尺寸变化
        # self.k_proj(x): [batch_size, seq_len, d_model] * [d_model, n_kv_heads * head_dim] -> [batch_size, seq_len, n_kv_heads * head_dim]
        # [batch_size, seq_len, n_kv_heads * head_dim] --view--> [batch_size, seq_len, n_kv_heads, head_dim] -> [batch_size, n_kv_heads, seq_len, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. 拓展维度
        # k shape:[batch_size, n_heads, seq_len, head_dim]
        # v shape:[batch_size, n_heads, seq_len, head_dim]
        k = self.repeat_kv(k, self.group_size)
        v = self.repeat_kv(v, self.group_size)

        # 3. 计算注意力分数，
        # q: [batch_size, n_heads, seq_len, head_dim] 
        # k^T: [batch_size, n_heads, seq_len, head_dim] -> [batch_size, n_heads, head_dim, seq_len]
        # q * k^T: [batch_size, n_heads, seq_len, head_dim] * [batch_size, n_heads, head_dim, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        # attn_scores shape: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -torch.inf)

        # 5.计算注意力的概率分布
        # attn_probas: [batch_size, n_heads, seq_len, seq_len]
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 6.计算注意力权重
        # attn_probas: [batch_size, n_heads, seq_len, seq_len]
        # v: [batch_size, n_heads, seq_len, head_dim]
        # attn_weight: [batch_size, n_heads, seq_len, head_dim]
        attn_weight = torch.matmul(attn_probs, v)

        # 7.多头注意力合并
        # attn_weight: [batch_size, n_heads, seq_len, head_dim]
        output = attn_weight.transpose(1,2).contiguous().view(batch_size,seq_len, self.n_heads * self.head_dim)

        # 8.输出结果
        return self.o_proj(output)


# GQA模块的实现
if __name__ == "__main__":
    # 示例代码
    model = GQA(
        d_model=128,
        n_heads=8,
        n_kv_heads=2,
    )

    x = torch.randn(2, 10, 128)

    output = model(x)

    print('x shape:',x.shape)
    print('output shape:',output.shape)
    print(model)



        



