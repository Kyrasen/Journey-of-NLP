# BERT

>模型主要参数

- L: the number of layers (i.e., Transformer blocks)
- H: hidden size
- A: the number of self-attention heads

## 1 Pre-Traning 预训练
### 1.1 两个主要任务
#### Masked LM (MLM)

>构建 MLM 数据集
- 对 `15%` 的数据进行以下处理
    - `80% of the time`: 用 `[MASK]` 替换掉原来的某个 token
    - `10% of the time`: 用一个 `随机的 token` 替换掉原来的 token
    - `10% of the time`: 保持 `原来的 token` 不变

#### Next Sentence Prediction (NSP)


>两个特殊标记符 Special Token
- `[CLS]`: 放在句首
- `[SEP]`: 放在两句之间和最后

>构建 NSP 数据集
- 构建 `[A句, B句]` 对 
- `50% of the time`: B 是 A 的真正的下一句
- `50% of the time`: B 是随机的一句话

### 1.2 Representation

<div align=center>
    <image src="../imgs/bert-representation.png" width=600>
</div>


### 1.3 Example
```sh
# 原数据
这少年便是闰土。我认识他时，也不过十多岁，离现在将有三十年了。

# Token Embeddings (后面还要过一个 nn.Embedding)
[CLS]这少年便是闰土[SEP]我认识他时[SEP]

# Segment Embeddings (后面还要过一个 nn.Embedding)
[CLS] 这 少 年 便 是 闰 土 [SEP] 我 认 识 他 时 [SEP]
0     0  0  0  0 0  0 0  0     1  1 1  1  1  1
# Position Embeddings (后面还要过一个 nn.Embedding)
[CLS] 这 少 年 便 是 闰 土 [SEP] 我 认 识  他  时 [SEP]
0     1  2  3 4  5 6  7  8     9 10 11  12 13 14
```

## 2 Fine-Tuning 微调
- 针对具体的领域或者业务做微调，再接下游的任务，例如：文本分类、NER 识别...




