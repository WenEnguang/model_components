import torch
import torch.nn as nn

class MHA_KVCache(nn.Module):
    def __init__(self, dim: int = 512, n_heads: int = 8, use_cache: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.use_cache = use_cache
        self.head_dim = dim // n_heads


        # Wq, Wk, Wv Wo四个矩阵
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # 存储历史的tokens
        self.history_seq = []

    def forward(self, q, k, v, past_key=None, past_value=None, mask=None):
        batch_size, seq_len, _ = q.shape
        print("Input shape:", q.shape)
        print('seq_len:',seq_len)

        if not self.use_cache:
            self.history_seq.append(q)
        
        # Q,k,v的映射
        q = self.q_proj(q)
        if self.use_cache:
            k = self.k_proj(k)
            v = self.v_proj(v)

        # 切分多头： shape：(batch_size, n_heads, seq_len, head_dim)
        q  = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_cache:
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

      
        if self.use_cache:
            # kvcache会在seq_len维度上去拼接past_key和past_value
            if past_key is not None and past_value is not None:
                k = torch.cat([past_key, k], dim=2)
                v = torch.cat([past_value, v], dim=2)
        else:
            # 不使用cache时，就是标准的多头注意力
            history_seq = torch.cat(self.history_seq, dim=1)
            k = self.k_proj(history_seq)
            v = self.v_proj(history_seq)
            k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 保存KV，用于下一次预测
        past_key_values = (k, v)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        #添加mask 掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        #添加softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        #计算注意力输出
        attn_output = torch.matmul(attn_weights, v)

        #合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        #输出映射
        output = self.o_proj(attn_output)

        return output, past_key_values

if __name__ == "__main__":
    # 超参数
    batch_size = 64
    seq_len = 10
    dim = 4096
    heads = 8
    user_kv = True

    N = 100

    # 模拟输入
    x = torch.randn(batch_size, seq_len, dim)

    #  构造mask
    mask = torch.full((1,1,seq_len,seq_len),True)
    mask = torch.triu(mask, diagonal=1)
    print("Mask:\n", mask)

    mha_kv_cache = MHA_KVCache(dim=dim, n_heads=heads, use_cache=user_kv)

    # prefill阶段
    output, past_key_values = mha_kv_cache(x, x, x, mask=mask)
    print("Prefill output shape:", output.shape,'past_key shape',past_key_values[0].shape)

    # decode阶段，输入一个token

    x = output

    for _ in range(N):
        new_token = output[:,[-1],:]  # 取最后一个token作为输入
        output, past_key_values = mha_kv_cache(
            new_token,
            new_token,
            new_token,
            past_key=past_key_values[0],
            past_value=past_key_values[1],
            mask=None,
        )
        print("Decode output shape:", output.shape,'past_key shape',past_key_values[0].shape)

