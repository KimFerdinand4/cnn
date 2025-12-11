"""
Transformer模型实现
A complete implementation of the Transformer architecture from "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x):
        """分割成多个注意力头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """合并多个注意力头"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分割成多头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 计算注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = self.combine_heads(attn_output)
        
        # 最后的线性变换
        output = self.W_o(output)
        
        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        """生成源序列和目标序列的mask"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)
        
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        """编码器前向传播"""
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """解码器前向传播"""
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """前向传播"""
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        
        return output


if __name__ == "__main__":
    # 测试代码
    print("测试Transformer模型...")
    
    # 设置参数
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_seq_len = 100
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # 生成随机输入
    src = torch.randint(1, src_vocab_size, (2, 10))  # batch_size=2, seq_len=10
    tgt = torch.randint(1, tgt_vocab_size, (2, 10))  # batch_size=2, seq_len=10
    
    # 前向传播
    print(f"\n输入形状: src={src.shape}, tgt={tgt.shape}")
    output = model(src, tgt)
    print(f"输出形状: {output.shape}")
    print(f"预期形状: (batch_size={src.shape[0]}, seq_len={tgt.shape[1]}, vocab_size={tgt_vocab_size})")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n✓ Transformer模型测试成功!")
