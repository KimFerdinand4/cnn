# Transformer模型实现

这是一个完整的、可运行的Transformer模型实现，基于论文 "Attention is All You Need"。

## 功能特点

- ✅ 完整的Transformer架构实现
- ✅ 多头注意力机制 (Multi-Head Attention)
- ✅ 位置编码 (Positional Encoding)
- ✅ 编码器-解码器结构
- ✅ 可配置的模型参数
- ✅ 包含多个使用示例
- ✅ 支持序列到序列任务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 测试Transformer模型

直接运行transformer.py来测试模型：

```bash
python transformer.py
```

### 2. 运行完整示例

运行包含多个使用场景的示例：

```bash
python example_transformer.py
```

## 模型架构

### 主要组件

1. **MultiHeadAttention**: 多头注意力机制
   - 支持自注意力和交叉注意力
   - 可配置的注意力头数量

2. **PositionalEncoding**: 位置编码
   - 使用正弦和余弦函数
   - 为序列添加位置信息

3. **EncoderLayer**: 编码器层
   - 自注意力 + 前馈网络
   - Layer Normalization和残差连接

4. **DecoderLayer**: 解码器层
   - 自注意力 + 交叉注意力 + 前馈网络
   - 支持掩码机制

5. **Transformer**: 完整模型
   - 可配置的编码器和解码器层数
   - 灵活的模型大小设置

## 使用示例

### 示例1: 基本使用

```python
from transformer import Transformer
import torch

# 创建模型
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)

# 准备输入数据
src = torch.randint(1, 5000, (2, 10))  # (batch_size, seq_len)
tgt = torch.randint(1, 5000, (2, 10))

# 前向传播
output = model(src, tgt)
print(output.shape)  # (2, 10, 5000)
```

### 示例2: 训练模型

```python
import torch.nn as nn
import torch.optim as optim

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练循环
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1])
    loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
```

### 示例3: 获取注意力权重

```python
from transformer import MultiHeadAttention

attention = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)

output, attention_weights = attention(x, x, x)
print(attention_weights.shape)  # (batch_size, num_heads, seq_len, seq_len)
```

## 模型参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| src_vocab_size | 源词汇表大小 | 必需 |
| tgt_vocab_size | 目标词汇表大小 | 必需 |
| d_model | 模型维度 | 512 |
| num_heads | 注意力头数 | 8 |
| num_encoder_layers | 编码器层数 | 6 |
| num_decoder_layers | 解码器层数 | 6 |
| d_ff | 前馈网络维度 | 2048 |
| max_seq_len | 最大序列长度 | 5000 |
| dropout | Dropout比率 | 0.1 |

## 预定义模型配置

### 小型模型
```python
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512
)
# 约 2.5M 参数
```

### 中型模型
```python
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=256,
    num_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024
)
# 约 16M 参数
```

### 大型模型 (类似原论文)
```python
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)
# 约 60M 参数
```

## 应用场景

1. **机器翻译**: 将一种语言翻译成另一种语言
2. **文本摘要**: 生成文本的简短摘要
3. **问答系统**: 根据问题生成答案
4. **对话系统**: 生成对话响应
5. **序列标注**: 命名实体识别、词性标注等
6. **时间序列预测**: 预测未来的序列值

## 文件说明

- `transformer.py`: Transformer模型的完整实现
- `example_transformer.py`: 包含5个不同使用场景的示例
- `requirements.txt`: 项目依赖
- `TRANSFORMER_README.md`: 本文档

## 技术细节

### 注意力机制

使用缩放点积注意力：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### 位置编码

使用正弦和余弦函数：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 掩码机制

- **源序列掩码**: 忽略填充token
- **目标序列掩码**: 防止看到未来的token（因果掩码）

## 性能优化建议

1. **使用GPU**: 将模型和数据移到GPU上
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

2. **混合精度训练**: 使用torch.cuda.amp加速训练
3. **梯度累积**: 在显存受限时模拟大批量训练
4. **学习率预热**: 使用学习率预热策略

## 参考文献

- Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS.

## 许可证

本实现仅供学习和研究使用。

## 贡献

欢迎提交问题和改进建议！
