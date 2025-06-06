import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from adaptive_softmax import AdaptiveSoftmax

class StandardSoftmaxModel(nn.Module):
    """标准Softmax模型"""
    def __init__(self, input_size, output_size, dropout=0.1):
        super(StandardSoftmaxModel, self).__init__()
        self.projection = nn.Linear(input_size, output_size)
        self.dropout = dropout
        
    def forward(self, inputs):
        if self.dropout > 0:
            inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        return self.projection(inputs)
    
    def log_prob(self, inputs, target):
        if self.dropout > 0:
            inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        logits = self.projection(inputs)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

def benchmark_models(vocab_sizes, input_size=512, batch_size=64):
    """比较不同词表大小下标准Softmax和自适应Softmax的性能"""
    results = {
        'vocab_sizes': vocab_sizes,
        'standard_forward_time': [],
        'adaptive_forward_time': [],
        'standard_backward_time': [],
        'adaptive_backward_time': [],
        'standard_memory': [],
        'adaptive_memory': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    for vocab_size in vocab_sizes:
        print(f"测试词表大小: {vocab_size}")
        
        # 创建标准Softmax模型
        standard_model = StandardSoftmaxModel(input_size, vocab_size).to(device)
        
        # 创建自适应Softmax模型
        # 根据词表大小设置cutoffs
        if vocab_size <= 10000:
            cutoffs = [2000, vocab_size]
        else:
            cutoffs = [2000, 10000, vocab_size]
        adaptive_model = AdaptiveSoftmax(input_size, cutoffs).to(device)
        
        # 创建随机输入和目标
        inputs = torch.randn(batch_size, input_size, device=device)
        targets = torch.randint(0, vocab_size, (batch_size,), device=device)
        
        # 测量标准Softmax前向传播时间
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        start_time = time.time()
        for _ in range(10):  # 运行多次取平均
            _ = standard_model(inputs)
        torch.cuda.synchronize()
        standard_forward_time = (time.time() - start_time) / 10
        
        # 测量标准Softmax内存使用
        standard_memory = torch.cuda.memory_allocated() - start_memory if torch.cuda.is_available() else 0
        
        # 测量标准Softmax反向传播时间
        start_time = time.time()
        for _ in range(10):
            logits = standard_model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
        torch.cuda.synchronize()
        standard_backward_time = (time.time() - start_time) / 10
        
        # 清除梯度
        standard_model.zero_grad()
        
        # 测量自适应Softmax前向传播时间
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        start_time = time.time()
        for _ in range(10):
            _ = adaptive_model(inputs)
        torch.cuda.synchronize()
        adaptive_forward_time = (time.time() - start_time) / 10
        
        # 测量自适应Softmax内存使用
        adaptive_memory = torch.cuda.memory_allocated() - start_memory if torch.cuda.is_available() else 0
        
        # 测量自适应Softmax反向传播时间
        start_time = time.time()
        for _ in range(10):
            log_probs = adaptive_model.log_prob(inputs, targets)
            loss = -log_probs.mean()
            loss.backward()
        torch.cuda.synchronize()
        adaptive_backward_time = (time.time() - start_time) / 10
        
        # 清除梯度
        for param in adaptive_model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # 存储结果
        results['standard_forward_time'].append(standard_forward_time)
        results['adaptive_forward_time'].append(adaptive_forward_time)
        results['standard_backward_time'].append(standard_backward_time)
        results['adaptive_backward_time'].append(adaptive_backward_time)
        results['standard_memory'].append(standard_memory / (1024 * 1024))  # 转换为MB
        results['adaptive_memory'].append(adaptive_memory / (1024 * 1024))  # 转换为MB
        
        print(f"标准Softmax - 前向时间: {standard_forward_time:.4f}s, 反向时间: {standard_backward_time:.4f}s, 内存: {standard_memory / (1024*1024):.2f}MB")
        print(f"自适应Softmax - 前向时间: {adaptive_forward_time:.4f}s, 反向时间: {adaptive_backward_time:.4f}s, 内存: {adaptive_memory / (1024*1024):.2f}MB")
        print(f"加速比 - 前向: {standard_forward_time / adaptive_forward_time:.2f}x, 反向: {standard_backward_time / adaptive_backward_time:.2f}x")
        print(f"内存节省: {(1 - adaptive_memory / standard_memory) * 100:.2f}%")
        print("-" * 50)
    
    return results

def plot_results(results):
    """绘制比较结果"""
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 负号显示
    
    # 绘制前向传播时间对比
    plt.subplot(2, 2, 1)
    plt.plot(results['vocab_sizes'], results['standard_forward_time'], 'o-', label='标准Softmax')
    plt.plot(results['vocab_sizes'], results['adaptive_forward_time'], 's-', label='自适应Softmax')
    plt.title('前向传播时间对比')
    plt.xlabel('词表大小')
    plt.ylabel('时间 (秒)')
    plt.legend()
    plt.grid(True)
    
    # 绘制反向传播时间对比
    plt.subplot(2, 2, 2)
    plt.plot(results['vocab_sizes'], results['standard_backward_time'], 'o-', label='标准Softmax')
    plt.plot(results['vocab_sizes'], results['adaptive_backward_time'], 's-', label='自适应Softmax')
    plt.title('反向传播时间对比')
    plt.xlabel('词表大小')
    plt.ylabel('时间 (秒)')
    plt.legend()
    plt.grid(True)
    
    # 绘制内存使用对比
    plt.subplot(2, 2, 3)
    plt.plot(results['vocab_sizes'], results['standard_memory'], 'o-', label='标准Softmax')
    plt.plot(results['vocab_sizes'], results['adaptive_memory'], 's-', label='自适应Softmax')
    plt.title('内存使用对比')
    plt.xlabel('词表大小')
    plt.ylabel('内存 (MB)')
    plt.legend()
    plt.grid(True)
    
    # 绘制加速比
    plt.subplot(2, 2, 4)
    forward_speedup = np.array(results['standard_forward_time']) / np.array(results['adaptive_forward_time'])
    backward_speedup = np.array(results['standard_backward_time']) / np.array(results['adaptive_backward_time'])
    memory_saving = (1 - np.array(results['adaptive_memory']) / np.array(results['standard_memory'])) * 100
    
    plt.plot(results['vocab_sizes'], forward_speedup, 'o-', label='前向加速比')
    plt.plot(results['vocab_sizes'], backward_speedup, 's-', label='反向加速比')
    plt.plot(results['vocab_sizes'], memory_saving / 100, '^-', label='内存节省比例')
    plt.title('性能提升对比')
    plt.xlabel('词表大小')
    plt.ylabel('倍数 / 比例')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('softmax_comparison.png')
    plt.show()

if __name__ == "__main__":
    # 测试不同词表大小
    vocab_sizes = [3000, 5000, 10000, 50000, 100000, 200000]
    
    print("开始性能基准测试...")
    results = benchmark_models(vocab_sizes)
    
    print("绘制结果...")
    plot_results(results)
    
    print("基准测试完成，结果已保存为 softmax_comparison.png") 