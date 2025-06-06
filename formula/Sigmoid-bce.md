# Logistics Regression 逻辑回归

## 1 Sigmoid函数
>Sigmoid 函数

$$
S(x)=\frac{1}{1+e^{-x}}
$$


>Sigmoid 导数

$$
S^{\prime}(x)=\frac{e^{-x}}{\left(1+e^{-x}\right)^2}=S(x)(1-S(x))
$$


## 2 BCE（Binary Cross Entropy）二元交叉熵
### 2.1 单条数据

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


### 2.2 Batch 条数据
- N = batch size
$$
\operatorname{Loss}=-\frac{1}{N} \sum_{i=1}^N y_i \cdot \log \left(p\left(y_i\right)\right)+\left(1-y_i\right) \cdot \log \left(1-p\left(y_i\right)\right)
$$


## 3 BCE & Sigmoid联合求导

$$\frac{\partial \text{BEC}}{\partial z} = p - y$$