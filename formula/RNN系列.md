# RNN系列


<div align=center>
    <image src="../第二阶段学习/pic/rnn.png" width=800>
</div>

## 1 RNN 循环神经网络

$$h_t = \text{tanh}(xW_x + h_{t-1}W_{h})$$
- 输出：
    - 最后一个隐藏状态 $h_t$: 可以用来做文本分类
    - outputs: 前面所有时间步（token）记录的 $h_t$。可以用来做文本生成、机器翻译、文本分类（需要进行一定的处理）

## 2 LSTM 门机制
- 门：值域：(0, 1), 表示某个信息的留存程度
>遗忘门

$$
f_t=\sigma(xU_f + h_{t-1}W_f)
$$

>输入门

$$
i_t=\sigma(xU_i + h_{t-1}W_i)
$$

>细胞状态
- 候选细胞状态
$$\hat{c}_t = \text{tanh}(xU_c + hW_c)$$

- 计算新的细胞状态

$$c_t = f_t \odot c_{t-1} i_t \odot \hat{c}$$


>输出门

$$
o_t=\sigma(xU_o + h_{t-1}W_o)
$$

$$h_t = o_t \text{tanh}(c_t)$$


## 3 GRU

>Reset Gate

$$r_t = \sigma(U_{r} x_t  + W_{r} h_{t-1})$$

>Update Gate

$$z_t = \sigma(U_{z} x_t + W_{z} h_{t-1})$$

>输出

$$c_t = \tanh(U_{n} x_t + r_t * (W_{n} h_{t-1}))$$

$$h_t = (1 - z_t) * c_t + z_t * h_{t-1}$$


## 4 Seq2Seq
- 翻译任务
- Encoder 和 Decoder 的结构，他们分别是一个 RNN
    - Encoder 对 src 进行 encode
        - 输出： $h_t$ 给 Decoder 用
    - Decoder 对 tgt 进行 decode

## 5 文本生成（RNN）
### 5.1 使用 RNN 的输出 outputs
- shape: (batch_size, seq_len, embedding_dim)
- 每次使用的是 seq_len 维度的最后一个


### 5.2 使用 RNN 的 $h_x$ 参数
- 将上一个 token 过 Model 出来的 hx 塞入到下一个 token 的 hx（包含在模型的 forward 函数里） 参数里面


### 5.3 生成的方式
1. 贪婪算法/贪心算法: 选最大概率的那个 token
2. top-k: 从前 k 个里面随机选一个
3. top-p: 累积概率超过 p 就丢弃，从剩下里面随机选一个
4. temperature: 用 $\text{softmax}(logits/temperature)$，按概率分布随机选一个