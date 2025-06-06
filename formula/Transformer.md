# Transformer 系列

## 1 Transformer
### 1.1 Multi-Head Attention
<div align=center>
    <image src="imgs/multiHeadAttention.png" width=500>
</div>


>Multi-Head Attention
```py
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # 进来做 projection, 并 reshape, 划分多头
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch_size, seq_len, num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, num_heads, self.d_k).transpose(1, 2)


        # 计算 attention scores
        attention_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)

        # 是否应用 mask
        if mask is not None:
            # 调整维度
            attention_scores = attention_scores.masked_fill(mask==0, -1e9)

        # softmax
        attention_scores = torch.softmax(attention_scores, dim=-1) # shape(batch_size, num_heads, seq_len, seq_len)


        # 与 V 相乘
        output = attention_scores @ V # shape(batch_size, num_heads, seq_len, d_k)

        # reshape 回去, 把多头合并回去
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_Len, d_model)

        # 最后输出的 projection

        output = self.o_proj(output)

        # dropout
        return self.dropout(output)
```

### 1.2 Layer Normalize
- 对 d_model 维度进行求均值和标准差
- 注意除零操作，需要加一个 esp=1e-5
- 可学习的参数
- 如果使用 nn.LayerNorm, 传入的初始化参数是 d_moel

### 1.3 Positional Encoding
$$
\begin{aligned}
P E_{(pos, 2 i)} & =\sin \left(pos / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(pos, 2 i+1)} & =\cos \left(pos / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$
- PE 的两个维度
    - pos: 取值范围 [0, seq_len]
    - 2i/2i+1: d_model 的奇数偶数位置
- 为了防止溢出，对 sin 和 cos 里面的项进行了一定的数学处理

## 2 GPT
- 使用 Transformer Decoder-Only 结构

## 3 BERT

### 3.1 两个 Task
#### 3.1.1 Pre-Training
##### 1 MLM
15% 的概率, 进行以下处理：
- 80% 概率: 用 [MASK] 替换 token（不包含特殊 token, 如：[CLS] [SEP]）
- 10% 概率: 用随机的 token 替换原来的 token
- 10% 概率: 保持原样

##### 2 NSP
- 正采样 50%：真实的下一句
- 负采样 50%：随机的一句

#### 3.1.2 Fine-Tuning
- 在特定的领域或者业务数据，基于原来的模型进行再训练

### 3.2 Embedding
<div align=center>
    <image src="imgs/bert-representation.png" width=800>
</div>

>Token Embedding
- 已经包含了 MLM 任务里面的 token

>Segment Embeddings
- 怎么划分前后句
- 注意 nn.Embedding 的初始化维度是 (2, d_model)

>Position Embeddings
- 注意 nn.Embedding 的初始化维度是（seq_len, d_model）


