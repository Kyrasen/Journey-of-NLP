# 1.25 Word2Vec 复习

Word2Vec
## 1 Word2Vec Intro
- Word2Vec 产生的动机：训练词向量，得到一个能够用来表示词的向量矩阵（支持词按索引得到每个词的向量表示）
- 两个模型
    - CBoW: 周围词预测中心词
    - Skip-gram: 中心词预测周围词
- 重点
    - 如何去构建数据集
    - 如何去构建任务

## 2 Raw Word2Vec
>任务的构建
- 多分类任务，类别为：len(vocab)
- 直观理解
    - 取某个词的向量
    - 然后跟一个矩阵（所有词的矩阵）做运算，得到 len(vocab) （类别）的一个值
    - 经过 softmax 转化成概率表示，这个概率的含义就是与 vocab 内每个词的相关程度
>数据集的构建
- 格式: (word, another_word)
- inputs: 
    - CBoW: word = 周围词
    - Skip-gram: word = 中心词
- labels: 
    - CBoW: another_word = 中心词
    - Skip-gram: another_word = 周围词

## 3 改进之后的版本
>任务的构建
- 二分类任务，类别为：0（不相关）；1（相关）
- 直观的理解
    - 取 word 的向量表示
    - 取 another_word 的向量表示
    - 两者做向量乘（相似度）的运算
    - 输出的值，经过 sigmoid 转化成 0-1 的一个值

>数据集的构建
- 格式：(word, another_word, label)
- CBoW:
    - input: word = 周围词，another_word = 中心词
    - label: label
- Skip-gram:
    - input: word = 中心词，another_word = 周围词
    - label: label







