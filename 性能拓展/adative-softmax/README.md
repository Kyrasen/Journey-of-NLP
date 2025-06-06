# 自适应Softmax (Adaptive Softmax) 实现

这个项目实现了自适应Softmax (Adaptive Softmax)，这是一种针对大型语言模型中大词表问题的优化技术。自适应Softmax能够显著加速大词表语言模型的训练和推理过程，同时降低内存使用量。

## 原理介绍

自适应Softmax的核心思想是**根据词频将词表分组**，对高频词进行完整计算，而对低频词使用层次化结构和降维技术，从而减少计算量。具体而言：

1. 将词表分为多个簇（clusters）：高频词、中频词和低频词
2. 对高频词使用全维度特征进行计算
3. 对中低频词先进行降维，然后再计算概率分布
4. 计算时只需要考虑相关簇，避免对整个词表进行计算

这种方法的优势在大词表场景下尤为明显，能够：
- 降低计算复杂度
- 减少内存使用
- 加速训练和推理过程
- 保持模型精度

## 项目文件结构

本项目包含以下文件：

- `adaptive_softmax.py`: 自适应Softmax的核心实现
- `language_model_with_adaptive_softmax.py`: 将自适应Softmax整合到语言模型中的示例
- `benchmark_softmax.py`: 用于比较标准Softmax和自适应Softmax性能的基准测试脚本
- `README.md`: 项目说明文档

## 使用方法

### 1. 安装依赖

```bash
pip install torch matplotlib numpy
```

### 2. 使用自适应Softmax

```python
import torch
from adaptive_softmax import AdaptiveSoftmax

# 参数设置
vocab_size = 100000  # 总词表大小
input_size = 512     # 输入特征维度
batch_size = 32      # 批次大小

# 根据词频划分词表
cutoffs = [2000, 10000, vocab_size]  # [高频词截止, 中频词截止, 总词表大小]

# 创建自适应Softmax层
adaptive_softmax = AdaptiveSoftmax(input_size=input_size, cutoffs=cutoffs)

# 前向传播计算
inputs = torch.randn(batch_size, input_size)
logits = adaptive_softmax(inputs)

# 计算损失（训练时使用）
targets = torch.randint(0, vocab_size, (batch_size,))
log_probs = adaptive_softmax.log_prob(inputs, targets)
loss = -log_probs.mean()
loss.backward()
```

### 3. 将自适应Softmax整合到语言模型中

请参考 `language_model_with_adaptive_softmax.py` 文件，其中展示了如何将自适应Softmax整合到基于LSTM的语言模型中。

### 4. 性能基准测试

运行以下命令，比较不同词表大小下标准Softmax和自适应Softmax的性能：

```bash
python benchmark_softmax.py
```

这将生成性能对比图表 `softmax_comparison.png`，包括：
- 前向传播时间对比
- 反向传播时间对比
- 内存使用对比
- 性能提升比例

## 调整参数

自适应Softmax的主要参数是`cutoffs`，它决定了词表的分组方式。根据您的具体词表大小和词频分布，可以调整这个参数以获得最佳性能。

一般来说，cutoffs的设置可以遵循以下原则：
- 高频词簇（第一个簇）包含约5%-20%的词表
- 中频词簇包含约20%-30%的词表
- 剩余的为低频词簇

## 注意事项

1. 自适应Softmax主要适用于大词表场景（通常词表大小>10,000）
2. 对于小词表，标准Softmax可能更为高效
3. 在GPU上使用时性能提升更为明显
4. 可以根据具体的词频分布调整cutoffs参数

## 参考文献

[1] Grave, E., Joulin, A., Cissé, M., Jégou, H., & others. (2017). Efficient softmax approximation for GPUs. In International Conference on Machine Learning (ICML). 