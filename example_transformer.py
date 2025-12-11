"""
Transformer使用示例
Examples of using the Transformer model for various tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import numpy as np


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("=" * 60)
    print("示例1: Transformer基本使用")
    print("=" * 60)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        max_seq_len=50
    )
    
    # 创建输入数据
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 10
    
    src = torch.randint(1, 1000, (batch_size, src_seq_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_seq_len))
    
    # 前向传播
    output = model(src, tgt)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_2_training_loop():
    """示例2: 简单的训练循环"""
    print("=" * 60)
    print("示例2: Transformer训练示例")
    print("=" * 60)
    
    # 模型参数
    vocab_size = 100
    d_model = 128
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_seq_len=50
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 生成一些虚拟数据
    num_samples = 10
    src_data = torch.randint(1, vocab_size, (num_samples, 15))
    tgt_data = torch.randint(1, vocab_size, (num_samples, 15))
    
    # 训练几个epoch
    model.train()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i in range(num_samples):
            # 获取单个样本
            src = src_data[i:i+1]
            tgt_input = tgt_data[i:i+1, :-1]
            tgt_output = tgt_data[i:i+1, 1:]
            
            # 前向传播
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            # 计算损失
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
    
    print("训练完成!")
    print()


def example_3_sequence_translation():
    """示例3: 序列到序列翻译示例"""
    print("=" * 60)
    print("示例3: 序列到序列翻译")
    print("=" * 60)
    
    # 简化的词汇表
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'hello': 3, 'world': 4, 'how': 5, 'are': 6, 'you': 7}
    tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '你好': 3, '世界': 4, '怎么': 5, '样': 6}
    
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_seq_len=20
    )
    
    # 示例句子: "hello world" -> "你好 世界"
    src_sentence = torch.tensor([[1, 3, 4, 2]])  # <sos> hello world <eos>
    tgt_sentence = torch.tensor([[1, 3, 4, 2]])  # <sos> 你好 世界 <eos>
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(src_sentence, tgt_sentence[:, :-1])
        predictions = torch.argmax(output, dim=-1)
    
    print(f"源句子ID: {src_sentence.squeeze().tolist()}")
    print(f"目标句子ID: {tgt_sentence.squeeze().tolist()}")
    print(f"预测ID: {predictions.squeeze().tolist()}")
    print()


def example_4_attention_visualization():
    """示例4: 注意力权重可视化准备"""
    print("=" * 60)
    print("示例4: 获取注意力权重")
    print("=" * 60)
    
    from transformer import MultiHeadAttention
    
    # 创建注意力层
    d_model = 128
    num_heads = 8
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 计算注意力
    output, attention_weights = attention(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重 (batch 0, head 0):")
    print(attention_weights[0, 0].detach().numpy())
    print()


def example_5_custom_model():
    """示例5: 自定义模型配置"""
    print("=" * 60)
    print("示例5: 自定义Transformer配置")
    print("=" * 60)
    
    # 创建不同大小的模型
    configs = [
        {"name": "小型模型", "d_model": 128, "num_heads": 4, "num_layers": 2, "d_ff": 512},
        {"name": "中型模型", "d_model": 256, "num_heads": 8, "num_layers": 4, "d_ff": 1024},
        {"name": "大型模型", "d_model": 512, "num_heads": 8, "num_layers": 6, "d_ff": 2048},
    ]
    
    for config in configs:
        model = Transformer(
            src_vocab_size=5000,
            tgt_vocab_size=5000,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            d_ff=config["d_ff"]
        )
        
        params = sum(p.numel() for p in model.parameters())
        print(f"{config['name']}: {params:,} 参数")
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Transformer模型使用示例集合")
    print("=" * 60 + "\n")
    
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_training_loop()
    example_3_sequence_translation()
    example_4_attention_visualization()
    example_5_custom_model()
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
