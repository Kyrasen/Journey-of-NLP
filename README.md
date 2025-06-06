# NLP学习
## 第一阶段：
  主要是入门NLP，了解NLP发展的历史，所做的项目有：
  - 线性回归：房价预测
  - 逻辑回归：葡萄酒分类、鸢尾花分类
### NLP
  - 文本分类：清华新闻数据集、头条数据集
### Word2Vec
  - 动机：想要训练一个词向量，能够对每个词进行编码
  - 重要假设：词的距离越近，越相关
  - 流程：首先，拿到一堆语料，然后构建一个此表（字典）vocab，用来索引Embedding，拿到每个词的向量表示；构建数据集（skip-gram）: win_size: 中心词两边的语境词（上下文词）；负采样：随机、基于词频的负采样

### Logistics Regression 逻辑回归

#### 1 Sigmoid函数
>Sigmoid 函数

$$
S(x)=\frac{1}{1+e^{-x}}
$$


>Sigmoid 导数

$$
S^{\prime}(x)=\frac{e^{-x}}{\left(1+e^{-x}\right)^2}=S(x)(1-S(x))
$$


#### 2 BCE（Binary Cross Entropy）二元交叉熵（这里无法正确显示公式，可参看formula/Sigmoid-bce.md文件
##### 2.1 单条数据

$$
\operatorname{Loss}=-[ y \cdot \log \left(p\left(y\right)\right)+\left(1-y\right) \cdot \log \left(1-p\left(y\right)\right)]
$$
- $y$: label
- $p(y)$: predict

猜硬币正反面的游戏，不是猜 0 或者 1, 猜 (0, 1) 之间的任意小数

>当答案是 1 的情况

$$
\operatorname{Loss}=- \log \left(p\left(y\right)\right)
$$
 
- 选手1：p(y) = 0.99999, Loss: ->0
- 选手2：p(y) = 0.5, Loss: 0.6931471805599453
- 选手3：p(y) = 0.5, Loss: 0.6931471805599453
- 选手4：p(y) = 0.49, Loss: 0.7133498878774648

>当答案是 0 的情况

$$
\operatorname{Loss}=- \log \left(1- p\left(y\right)\right)
$$
 

- 选手1：p(y) = 1e-5, Loss: 1.0000050000287824e-05
- 选手2：p(y) = 0.5, Loss: 0.6931471805599453
- 选手3：p(y) = 0.5, Loss: 0.6931471805599453
- 选手4：p(y) = 0.51, Loss: 0.7133498878774648


##### 2.2 Batch 条数据
- N = batch size
$$
\operatorname{Loss}=-\frac{1}{N} \sum_{i=1}^N y_i \cdot \log \left(p\left(y_i\right)\right)+\left(1-y_i\right) \cdot \log \left(1-p\left(y_i\right)\right)
$$


##### 3 BCE & Sigmoid联合求导

$$\frac{\partial \text{BEC}}{\partial z} = p - y$$

## 第二阶段
### 分词（Tokenization）
将文本、语料转换成token（按字分、按词分），输出
- vocab: 在encode（编码）的过程中使用
- idx2word：在decode（解码）的过程中使用
- OOV（out of vocab）:不在此表内；[UNK]记录一些不在此表内的词

### Embedding
过程：
1. 将token转换成向量化表示：从vocab拿到token的对应的id, 用id对某个Embedding进行索引，拿到token的向量表示
2. Embedding种类：
   - One-Hot:稀疏向量（矩阵）会浪费存储空间；
   - Random Vector: 固定的、随机的向量
   - nn.Embedding：PyTorch里面可学习的Embedding

### 经典模型复现：RNN, LSTM, GRU, Transformer, GPT, Bert，详情请参看“第二阶段学习”

## 第三阶段
RAG项目，详情请参看“美食RAG/readme.txt”文件
   

    
