import torch
import torch.nn as nn
from adaptive_softmax import AdaptiveSoftmax

class LanguageModelWithAdaptiveSoftmax(nn.Module):
    """
    使用自适应Softmax的简单语言模型
    
    该模型展示了如何将自适应Softmax整合到语言模型中，以提高大词表下的计算效率
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 num_layers=2, 
                 dropout=0.2):
        super(LanguageModelWithAdaptiveSoftmax, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 自适应Softmax输出层
        # 根据词频将词表分为多个部分
        # 例如：[2000, 10000, vocab_size] 表示：
        #   - 前2000个词为高频词
        #   - 2000-10000之间为中频词
        #   - 10000以上为低频词
        cutoffs = [2000, 10000, vocab_size]
        self.decoder = AdaptiveSoftmax(
            input_size=hidden_dim,
            cutoffs=cutoffs,
            dropout=dropout
        )
        
    def forward(self, x, targets=None):
        """
        前向传播
        
        参数:
            x: 输入序列，形状为 [batch_size, sequence_length]
            targets: 目标词索引，形状为 [batch_size, sequence_length]
            
        返回:
            如果提供targets，返回损失值；否则返回logits
        """
        # 获取词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 通过LSTM层
        lstm_output, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # 应用dropout
        lstm_output = self.dropout(lstm_output)
        
        # 重塑输出以便于解码
        batch_size, seq_len, _ = lstm_output.shape
        lstm_output = lstm_output.reshape(-1, self.hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # 如果提供了目标，计算损失
        if targets is not None:
            # 重塑目标
            targets = targets.reshape(-1)  # [batch_size * seq_len]
            
            # 使用自适应Softmax计算对数概率
            log_probs = self.decoder.log_prob(lstm_output, targets)
            
            # 计算损失（负对数似然）
            loss = -log_probs.mean()
            return loss
        else:
            # 否则返回所有词的logits
            logits = self.decoder(lstm_output)  # [batch_size * seq_len, vocab_size]
            return logits.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, vocab_size]
            
    def generate(self, start_tokens, max_length=100, temperature=1.0):
        """
        生成文本
        
        参数:
            start_tokens: 起始词索引，形状为 [batch_size, sequence_length]
            max_length: 生成的最大长度
            temperature: 采样温度，较低的值使分布更加尖锐
            
        返回:
            生成的词索引序列
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            batch_size = start_tokens.size(0)
            current_tokens = start_tokens.clone()
            
            # 初始化隐藏状态
            hidden = None
            
            for _ in range(max_length):
                # 获取当前输入的最后一个词
                inputs = current_tokens[:, -1].unsqueeze(1)  # [batch_size, 1]
                
                # 获取词嵌入
                embedded = self.embedding(inputs)  # [batch_size, 1, embedding_dim]
                
                # 通过LSTM层
                lstm_output, hidden = self.lstm(embedded, hidden)  # [batch_size, 1, hidden_dim]
                
                # 应用dropout
                lstm_output = self.dropout(lstm_output)
                
                # 重塑输出以便于解码
                lstm_output = lstm_output.reshape(-1, self.hidden_dim)  # [batch_size, hidden_dim]
                
                # 计算下一个词的logits
                logits = self.decoder(lstm_output)  # [batch_size, vocab_size]
                
                # 应用温度
                if temperature != 1.0:
                    logits = logits / temperature
                
                # 使用softmax获取概率分布
                probs = torch.softmax(logits, dim=-1)
                
                # 采样下一个词
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # 将新词添加到序列中
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
        return current_tokens

# 使用示例
def train_language_model_example():
    # 参数设置
    vocab_size = 100000  # 词表大小
    embedding_dim = 300  # 词嵌入维度
    hidden_dim = 512     # 隐藏层维度
    batch_size = 32      # 批次大小
    seq_length = 50      # 序列长度
    
    # 创建语言模型
    model = LanguageModelWithAdaptiveSoftmax(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 假设我们有一些训练数据
    # 实际应用中，这些数据会来自您的数据加载器
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # 训练模型
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    loss = model(inputs, targets)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"训练损失: {loss.item()}")
    
    # 文本生成示例
    model.eval()
    start_sequence = torch.randint(0, vocab_size, (2, 5))  # 2个样本，每个5个词
    generated = model.generate(start_sequence, max_length=10)
    
    print(f"生成序列形状: {generated.shape}")
    
    return model

if __name__ == "__main__":
    print("训练带有自适应Softmax的语言模型...")
    model = train_language_model_example()
    print("模型结构:")
    print(model) 