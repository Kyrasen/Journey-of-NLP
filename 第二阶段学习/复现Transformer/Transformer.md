# Transformer

- 组成组件：Embedding, PositionalEncoding, Multi-Head Attention, LayerNorm, FeedForward, Linear, Softmax
- 主要组件：PositionalEncoding, Multi-Head Attention, LayerNorm



## 1 Embedding

<div align=center>
    <image src="imgs/embedding.png" width=800>
</div>

## 2 Positional Encoding
- Embedding: output size (batch, seq_len, d_model)
$$
\begin{aligned}
P E_{(p o s, 2 i)} & =\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} & =\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$
- `pos`: seq_len 的位置，每个 token 在整个序列里面的位置
- `2i, 2i+1`: d_model 维度的偶数、奇数

>做一下数学变换

$$
\text{exp}^{\text{ln}(p o s / 10000^{2 i / d_{\text {model }}})}
$$

$$
\text{exp}^{\text{ln}(pos) - (2i/d_{model})\text{ln} (10000)}
$$

$$
\text{exp}^{\text{ln}(pos)} \cdot \text{exp}^{- 2i/d_{model}\text{ln} (10000)}
$$

$$
pos \cdot \text{exp}^{- 2i\cdot\text{ln} (10000)/d_{model}}
$$


## 3 Multi-Head Attention

>Scaled Dot Product Attention

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

>Multi-Head Attention

$$
\begin{aligned}
\operatorname{MultiHead}(Q, K, V) & =\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \text { head }_{\mathrm{h}}\right) W^O \\
\text { where head } & =\operatorname{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)
\end{aligned}
$$

- $W_i^Q \in \mathbb{R}^{d_{\text {model }} \times d_k}, W_i^K \in \mathbb{R}^{d_{\text {model }} \times d_k}, W_i^V \in \mathbb{R}^{d_{\text {model }} \times d_v}$ and $W^O \in \mathbb{R}^{h d_v \times d_{\text {model }}}$.

- $h=8$: heads 的数量
- $d_k=d_v=d_{\text {model }} / h=64$

## 4 Layer Normalize

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} \cdot \gamma+\beta
$$

>归一化部分

$$\mu=\frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2=\frac{1}{d} \sum_{i=1}^d\left(x_i-\mu\right)^2$$
- $\mu$ ：当前词向量的均值
- $\sigma^2$ ：当前词向量的方差
-  $\boldsymbol{\epsilon}$ ：防止除以零的小值（如 $e^{-5}$ ）

>缩放和平移部分

$$\hat{x}_i=\gamma \text{归一化后的值} +\beta$$

- $\gamma, \beta$ ：可学习的参数，用于调整归一化后的值。

## 5 Feed Forward

$$
\operatorname{FFN}(x)=\max \left(0, x W_1+b_1\right) W_2+b_2
$$

- `dmodel = 512`: input and output dimension
- `dff = 2048`: inner-layer dimension

## 6 Transformer 中的 Mask
### 6.1 Padding Mask
- 用在 encoder, decoder，为了忽略 PAD 位置


### 6.2 Causal Mask
- 用在 decoder
- 又称：sequence mask、future mask

<div align=center>
    <image src="imgs/causal_mask.png" width=300>
</div>

$$
p_i=\frac{e^{y_i}}{\sum_{k=1}^n e^{y_k}}
$$