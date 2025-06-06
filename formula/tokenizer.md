# Intro

## 1 分词（Tokenization）
将文本、语料转换成 token（按字分、按词分），输出
- vocab: 在 encode（编码） 的过程使用
- idx2word: 在 decode（解码）的过程使用

>OOV（Out of Vocabulary）
- 不在词表内
- `[UNK]`记录一些不在词表内的词

## 2 Embedding

### 2.1 过程
>将 token 转换成向量化表示
1. 从 vocab 拿到 token 的对应 id
2. 用 id 对 `某个Embedding` 进行索引，拿到 token 的向量表示

### 2.2 Embedding 种类
1. One-Hot: 稀疏向量（矩阵）会浪费存储空间；
2. Random Vector: 固定的、随机的向量
3. nn.Embedding: PyTorch 里面可学习的 Embedding











