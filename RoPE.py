'''
Positional Encoding:
    用于为序列中的每个位置添加位置信息，使得模型能够理解序列的顺序。
'''

# step:1、生成旋转频率

import torch
import torch.nn as nn
import numpy as np

def get_rotary_frequency(head_dim, seq_len, theta=1e5):
    '''
    生成RoPE的旋转频率
    Args:
        head_dim (int): 头维度
        seq_len (int): 序列长度
        theta (float): 旋转频率的参数
    Returns:
        freqs: shape (seq_len, dim // 2)，每个位置每个维度对的频率
    '''
    # 计算每个维度的基础频率
    # theta_i = 10000^(-2i/d), i = 0, 1, ..., d/2-1
    i = torch.arange(0, head_dim // 2, dtype=torch.float32)
    freqs = theta ** (-2 * i / head_dim)  # shape (head_dim // 2,)

    # 生成位置索引
    positions = torch.arange(0, seq_len, dtype=torch.float32)  # shape (seq_len,)

    # 计算每一个位置的角度： position * freqs
    angles = torch.outer(positions, freqs)  # shape (seq_len, head_dim // 2)

    return angles

# step:2、构建cos/sin缓存

def get_rotary_embedding(head_dim, seq_len, theta=1e5):
    '''
    预计算RoPE的cos和sin值
    Args:
        head_dim (int): 头维度
        seq_len (int): 序列长度
        theta (float): 旋转频率的参数
    Returns:
        cos: shape (seq_len, head_dim // 2)
        sin: shape (seq_len, head_dim // 2)
    '''
    angles = get_rotary_frequency(head_dim, seq_len, theta)
    # 计算cos和sin值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 将 (seq_len, dim//2) 扩展为 (seq_len, dim)，与 rotate_half 配合使用
    cos = torch.cat([cos,cos],dim=-1)
    sin = torch.cat([sin,sin],dim=-1)

    return cos, sin

# step:3、应用旋转变换

def rotate_half(x):
    """
    将向量的前半部分和后半部分交换，并对后半部分取负
    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]

    这是实现旋转的关键辅助函数
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q,k, cos, sin):
    """
    应用 RoPE 旋转变换（LLaMA 风格实现）

    Args:
        q: Query，shape (batch, seq_len, num_heads, head_dim)
        k: Key，shape (batch, seq_len, num_heads, head_dim)
        cos: shape (seq_len, head_dim)
        sin: shape (seq_len, head_dim)

    Returns:
        q_rot, k_rot: 旋转后的 Query 和 Key

    旋转公式：
        q' = q * cos + rotate_half(q) * sin
        k' = k * cos + rotate_half(k) * sin
    """
    # 调整 cos/sin 形状以便广播: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

# 完整的RoPE模块实现
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim:int, seq_len:int, theta:float=1e5):
        '''
        Args:
            head_dim (int): 注意力头维度
            seq_len (int): 序列长度
            theta (float): 旋转频率的参数
        '''
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.theta = theta

        # 预计算并缓存cos和sin值
        cos, sin = get_rotary_embedding(head_dim, seq_len, theta)
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self,q:torch.Tensor,k:torch.Tensor,position:torch.Tensor=None):
        """
        对 Query 和 Key 应用 RoPE

        Args:
            q: Query，shape (batch, seq_len, num_heads, head_dim)
            k: Key，shape (batch, seq_len, num_heads, head_dim)
            positions: 位置索引，默认为 [0, 1, 2, ..., seq_len-1]

        Returns:
            q_rot, k_rot: 旋转后的 Query 和 Key
        """
        seq_len = q.shape[1]

        # 获取当前序列长度的cos/sin
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]

        # 应用旋转
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        return q_rot, k_rot

# 测试
rope = RotaryPositionalEmbedding(head_dim=64, seq_len=4096)

# 模拟输入
batch_size = 2
seq_len = 128
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)

q_rot, k_rot = rope(q, k)
print(f"Q_rot shape: {q_rot.shape}")
print(f"K_rot shape: {k_rot.shape}")