import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveSoftmax(nn.Module):
    """
    自适应Softmax实现，针对大型语言模型中大词表问题的优化技术
    
    参数:
        input_size: 输入特征的维度
        cutoffs: 词表切分的边界列表，如 [2000, 10000, vocab_size]
        dropout: dropout概率
    """
    def __init__(self, input_size, cutoffs, dropout=0.1):
        super(AdaptiveSoftmax, self).__init__()
        
        self.input_size = input_size
        self.cutoffs = cutoffs
       
        self.output_size = cutoffs[-1]
        self.dropout = dropout
        
        # 计算每个簇的大小
        self.cutoff_ends = [0] + cutoffs
        self.cluster_sizes = []

        print(self.cutoff_ends)
        print(self.cutoffs)
        
        for i in range(len(self.cutoffs)):
            self.cluster_sizes.append(self.cutoff_ends[i+1] - self.cutoff_ends[i])
            
        
        # 维度缩减因子，对罕见词使用更低的维度
        self.shortlist_size = self.cluster_sizes[0]  # 高频词
        self.n_clusters = len(self.cluster_sizes) - 1  # 簇的数量（不包括高频词簇）
        
        # 定义高频词簇的全维度投影
        self.head_proj = nn.Linear(input_size, self.shortlist_size, bias=False)
        
        # 为各个簇分配不同的降维因子
        self.tail_projs = nn.ModuleList()
        self.tail_biases = nn.ParameterList()
        self.tail_dims = []
        
        for i in range(self.n_clusters):
            # 对低频词使用降维投影，维度随频率递减
            dim_factor = 2 if i == 0 else 4
            projection_dim = input_size // dim_factor
            self.tail_dims.append(projection_dim)
            
            # 两阶段投影：先降维，再投影到对应簇的大小
           
            self.tail_projs.append(nn.Sequential(
                nn.Linear(input_size, projection_dim, bias=False),
                nn.Linear(projection_dim, self.cluster_sizes[i+1], bias=False)
            ))
            
            self.tail_biases.append(nn.Parameter(torch.zeros(self.cluster_sizes[i+1])))
            
    def forward(self, inputs):
        """
        前向传播计算
        
        参数:
            inputs: 形状为 [batch_size, input_size] 的输入张量
            
        返回:
            outputs: 针对整个词表的logits
        """
        if self.dropout > 0:
            inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        
        # 计算头部高频词的logits (全维度计算)
        head_logits = self.head_proj(inputs)
        
        # 初始化输出logits
        batch_size = inputs.size(0)
        outputs = torch.zeros(batch_size, self.output_size, device=inputs.device)
        outputs[:, :self.shortlist_size] = head_logits
        
        # 计算各个低频词簇的logits
        for i in range(self.n_clusters):
            start_idx = self.cutoff_ends[i+1]
            end_idx = self.cutoff_ends[i+2]
            
            # 应用降维投影
            cluster_logits = self.tail_projs[i](inputs) + self.tail_biases[i]
            outputs[:, start_idx:end_idx] = cluster_logits
            
        return outputs
    
    def log_prob(self, inputs, target):
        """
        计算log概率，用于训练时的高效计算
        
        参数:
            inputs: 形状为 [batch_size, input_size] 的输入张量
            target: 形状为 [batch_size] 的目标词索引
            
        返回:
            log_probs: 每个目标词的对数概率
        """
        if self.dropout > 0:
            inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        
        batch_size = inputs.size(0)
        log_probs = torch.zeros(batch_size, device=inputs.device)
        
        # 将目标按照簇进行分组
        target_clusters = [[] for _ in range(len(self.cutoffs))]
        target_indices = [[] for _ in range(len(self.cutoffs))]
        batch_indices = [[] for _ in range(len(self.cutoffs))]
        
        # 将每个目标分配到对应的簇
        for batch_idx, t in enumerate(target):
            for cluster_idx in range(len(self.cutoffs)):
                if t < self.cutoff_ends[cluster_idx+1]:
                    target_clusters[cluster_idx].append(batch_idx)
                    # 如果是第一个簇，直接使用索引，否则需要减去簇的起始索引
                    relative_idx = t if cluster_idx == 0 else t - self.cutoff_ends[cluster_idx]
                    target_indices[cluster_idx].append(relative_idx)
                    batch_indices[cluster_idx].append(batch_idx)
                    break
        
        # 处理高频词簇
        if len(target_clusters[0]) > 0:
            batch_idx_tensor = torch.tensor(batch_indices[0], device=inputs.device)
            target_idx_tensor = torch.tensor(target_indices[0], device=inputs.device)
            
            head_logits = self.head_proj(inputs[batch_idx_tensor])
            
            # 计算头部的log_softmax
            head_logprob = F.log_softmax(head_logits, dim=1)
            log_probs[batch_idx_tensor] = head_logprob.gather(1, target_idx_tensor.unsqueeze(1)).squeeze(1)
        
        # 处理各个低频词簇
        for cluster_idx in range(1, len(self.cutoffs)):
            if len(target_clusters[cluster_idx]) > 0:
                batch_idx_tensor = torch.tensor(batch_indices[cluster_idx], device=inputs.device)
                target_idx_tensor = torch.tensor(target_indices[cluster_idx], device=inputs.device)
                
                # 计算该簇的logits
                cluster_logits = self.tail_projs[cluster_idx-1](inputs[batch_idx_tensor]) + self.tail_biases[cluster_idx-1]
                
                # 计算该簇的log_softmax
                cluster_logprob = F.log_softmax(cluster_logits, dim=1)
                log_probs[batch_idx_tensor] = cluster_logprob.gather(1, target_idx_tensor.unsqueeze(1)).squeeze(1)
        
        return log_probs

# 使用示例
def adaptive_softmax_example():
    # 参数设置
    vocab_size = 100000  # 总词表大小
    input_size = 512     # 输入特征维度
    batch_size = 32      # 批次大小
    
    # 根据词频划分词表（示例划分）
    cutoffs = [2000, 10000, vocab_size]  # 将词表分为3个部分：前2000个高频词，2000-10000中频词，其余低频词
    
    # 创建自适应Softmax模型
    adaptive_softmax = AdaptiveSoftmax(input_size=input_size, cutoffs=cutoffs)
    
    # 随机生成一批输入特征
    inputs = torch.randn(batch_size, input_size)
    
    # 随机生成目标词索引
    target = torch.randint(0, vocab_size, (batch_size,))
    
    # 计算log概率
    log_probs = adaptive_softmax.log_prob(inputs, target)
    print(f"Log概率形状: {log_probs.shape}")
    
    # 计算所有词的logits
    logits = adaptive_softmax(inputs)
    print(f"完整logits形状: {logits.shape}")
    
    # 计算损失
    loss = -log_probs.mean()
    print(f"损失值: {loss.item()}")
    
    return adaptive_softmax

if __name__ == "__main__":
    model = adaptive_softmax_example()
    print("自适应Softmax模型结构:")
    print(model) 