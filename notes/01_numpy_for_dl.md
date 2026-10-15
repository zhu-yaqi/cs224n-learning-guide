# Day 1：面向深度学习的 NumPy 基础

> CS224N 学习记录 — Day 1
> 核心：Shape、Axis、Broadcasting、Vectorization、Softmax、Cross Entropy、神经网络基础运算、SVD

---

## 1. 为什么学深度学习之前要学 NumPy？

今天最大的收获不是记住多少 NumPy API，而是开始建立：

> **Shape-first Thinking（先看 Shape 的思维）**

以后看到一个 NumPy / PyTorch 运算，我应该先问：

1. 输入的 Shape 是什么？
2. 每一个维度分别代表什么？
3. 运算发生在哪个 Axis？
4. 哪个维度会被消掉？
5. 是否发生 Broadcasting？
6. 矩阵乘法的维度是否匹配？
7. 最终输出的 Shape 是什么？

以后 Transformer 中会大量出现：

```text
[B, T, D]
```

通常表示：

* $B$：Batch Size，批次大小
* $T$：Sequence Length / Token 数量
* $D$：Feature / Embedding Dimension，特征维度

从现在开始养成一个习惯：

> **代码运行之前，先预测输出 Shape。**

---

# 2. Shape 与 ndim

例如：

```python
import numpy as np

a = np.arange(15).reshape(3, 5)
```

此时：

```python
a.shape
# (3, 5)

a.ndim
# 2
```

`shape = (3, 5)` 表示：

```text
3 行
5 列
```

为什么：

```python
a.ndim
# 2
```

因为定位一个元素需要两个坐标：

```python
a[row, column]
```

例如：

```python
a[1, 3]
```

需要同时知道：

```text
第几个 row
+
第几个 column
```

所以可以建立一个很实用的直觉：

> **ndim ≈ 定位一个元素需要几个坐标。**

---

## 2.1 `(3,)` 和 `(1, 3)` 不一样

例如：

```python
b = np.array([6, 7, 8])
```

它的 Shape：

```python
b.shape
# (3,)

b.ndim
# 1
```

它是一个一维数组。

但是：

```python
c = np.array([[6, 7, 8]])
```

它的 Shape：

```python
c.shape
# (1, 3)

c.ndim
# 2
```

所以：

```text
(3,) ≠ (1, 3)
```

这个区别在后面的 Broadcasting 中非常重要。

---

# 3. Axis：到底是什么意思？

假设：

```text
X.shape = (3, 4)
```

可以理解成：

```text
shape = (3, 4)
         ↑  ↑
         0  1
```

Shape 中的每一个位置都对应一个 Axis：

```text
axis=0 → 第 0 个维度
axis=1 → 第 1 个维度
```

今天最重要的理解是：

> **Reduction(axis=n) 会把第 n 个维度压缩掉。**

例如：

```python
X.sum(axis=0)
```

Shape 变化：

```text
(3, 4)
 ↑
axis=0 被压缩

→ (4,)
```

而：

```python
X.sum(axis=1)
```

Shape 变化：

```text
(3, 4)
    ↑
axis=1 被压缩

→ (3,)
```

不要死记：

```text
axis=0 = 列
axis=1 = 行
```

二维数组时这样理解暂时没问题，但进入深度学习后经常遇到三维甚至更高维 Tensor。

更好的方法是：

> **看 Shape → 找到 Axis → 想象把那个维度消掉。**

---

# 4. 用 `[B, T, D]` 理解 Axis

假设：

```text
X.shape = [B, T, D]
```

其中：

* $B$：Batch
* $T$：Tokens
* $D$：Features

那么：

```python
X.sum(axis=0)
```

相当于：

```text
[B, T, D]
 ↑
消掉 B

→ [T, D]
```

而：

```python
X.sum(axis=1)
```

相当于：

```text
[B, T, D]
    ↑
消掉 T

→ [B, D]
```

而：

```python
X.sum(axis=2)
```

相当于：

```text
[B, T, D]
       ↑
消掉 D

→ [B, T]
```

另外：

```python
X.sum(axis=-1)
```

表示：

> 对最后一个维度进行 Reduction。

因此对于：

```text
[B, T, D]
```

有：

```text
axis=-1

→ [B, T]
```

以后在深度学习代码中，`axis=-1` 或 PyTorch 中的 `dim=-1` 会非常常见。

---

# 5. `keepdims=True`

正常情况下：

```python
X.sum(axis=1)
```

会直接把 Axis 1 删除。

例如：

```text
X.shape = (3, 4)

X.sum(axis=1)

→ (3,)
```

但是：

```python
X.sum(axis=1, keepdims=True)
```

得到：

```text
(3, 1)
```

也就是说：

> `keepdims=True` 不彻底删除这个维度，而是把这个维度保留成大小为 $1$ 的维度。

例如：

```text
(3, 4)
    ↑
 axis=1

普通 Reduction：

→ (3,)

keepdims=True：

→ (3, 1)
```

这个 $1$ 非常重要，因为它经常用于后续的 Broadcasting。

---

# 6. Broadcasting（广播）

Broadcasting 允许 Shape 不完全相同、但满足兼容规则的数组进行逐元素运算。

例如：

```text
X.shape = (32, 128)
b.shape = (128,)
```

可以直接：

```python
X + b
```

最终 Shape：

```text
(32, 128)
```

---

## 6.1 Broadcasting 的核心规则

最重要的一句话：

> **从 Shape 最右边开始比较。**

两个维度满足以下条件之一，就可以兼容：

1. 两个维度相同；
2. 其中一个维度为 $1$；
3. 某一边缺少这个维度。

---

## 6.2 缺少的维度补在哪里？

假设：

```text
A.shape = (5, 7, 10)
b.shape =       (10,)
```

NumPy 从右边开始对齐：

```text
A: (5, 7, 10)
b: (      10)
```

缺少的维度可以理解成在左边补 $1$：

```text
A: (5, 7, 10)
b: (1, 1, 10)
```

然后逐维比较：

```text
10 vs 10 → 相同 ✓

7 vs 1 → 1 可以扩展成 7 ✓

5 vs 1 → 1 可以扩展成 5 ✓
```

所以：

```text
(5, 7, 10)
+
(10,)

→ (5, 7, 10)
```

---

## 6.3 「补 1」和「扩展」不是一回事

例如：

```text
b.shape = (10,)
```

不是直接补成：

```text
(5, 7, 10)
```

而是：

```text
第一步：

(10,)

↓

(1, 1, 10)
```

然后 Broadcasting：

```text
第二步：

(1, 1, 10)

↓

(5, 7, 10)
```

所以：

> **缺少的维度先视为 $1$，然后 Broadcasting 再把 $1$ 扩展到需要的大小。**

---

## 6.4 Broadcasting 失败的情况

例如：

```text
A.shape = (8, 3, 6, 20)
b.shape =       (3, 20)
```

右对齐：

```text
A: (8, 3, 6, 20)
b: (1, 1, 3, 20)
```

从右边比较：

```text
20 vs 20 → ✓

6 vs 3 → ✗
```

因为：

$6 \neq 3$ 并且：

$6 \neq 1,\qquad 3 \neq 1$ 所以 Broadcasting 失败。

---

# 7. `None`：手动增加一个维度

假设：

```python
x.shape
# (2,)
```

那么：

```python
x[:, None]
```

得到：

```text
(2, 1)
```

可以理解成：

```text
原来：

[x₀, x₁]

变成：

[[x₀],
 [x₁]]
```

而：

```python
x[None, :]
```

得到：

```text
(1, 2)
```

也就是：

```text
[[x₀, x₁]]
```

需要特别注意：

```python
X[:, 0]
```

和：

```python
X[:, None]
```

完全不是一个意思。

```text
[:, 0]     → 取第 0 列

[:, 1]     → 取第 1 列

[:, None]  → 增加一个维度
```

---

# 8. 矩阵乘法与 Shape

矩阵乘法最重要的 Shape 规则：

$[A,B] \times [B,C] \rightarrow [A,C]$ 中间两个维度必须相同。

例如：

```text
X.shape = (4, 3)
W.shape = (3, 2)
```

那么：

```python
X @ W
```

可以计算，因为：

$(4,3) \times (3,2)$ 中间维度满足：

$3 = 3$ 所以输出 Shape：

$(4,2)$ 这个模式以后会反复出现：

$[B,D] \times [D,H] \rightarrow [B,H]$---

# 9. Lab 1：Pairwise Cosine Similarity

给定：

```text
X: [N, D]
Y: [M, D]
```

目标是计算：

> $X$ 中每一个向量与 $Y$ 中每一个向量之间的 Cosine Similarity。

最终：

```text
similarity.shape = [N, M]
```

并且不使用 Python 双重 `for` 循环。

---

## 9.1 Cosine Similarity 公式

对于两个向量 $\mathbf{x}$ 和 $\mathbf{y}$：

$\operatorname{cosine}(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^{\mathsf T}\mathbf{y}} {\|\mathbf{x}\|_2\|\mathbf{y}\|_2}$ 其中 L2 Norm 为：

$\|\mathbf{x}\|_2 = \sqrt{\sum_{k=1}^{D}x_k^2}$---

## 9.2 一次计算所有 Dot Products

已知：

```text
X:   [N, D]

Y:   [M, D]

Y.T: [D, M]
```

所以：

```python
dot_products = X @ Y.T
```

Shape：

$[N,D] \times [D,M] \rightarrow [N,M]$ 矩阵中的第 $(i,j)$ 个元素：

$(\mathbf{X}\mathbf{Y}^{\mathsf T})_{ij} = \mathbf{x}_i^{\mathsf T}\mathbf{y}_j$ 也就是说，它保存了：

> 第 $i$ 个 $X$ 向量与第 $j$ 个 $Y$ 向量之间的点积。

这样就不需要：

```python
for x in X:
    for y in Y:
        ...
```

---

## 9.3 计算每个向量的 Norm

对于：

```text
X.shape = [N, D]
```

每一行代表一个向量。

所以：

```python
X_norms = np.sqrt(
    np.sum(X ** 2, axis=1)
)
```

对应数学公式：

$\|\mathbf{x}_i\|_2 = \sqrt{ \sum_{k=1}^{D} X_{ik}^2 }$ Shape 变化：

```text
[N, D]

axis=1

↓

[N]
```

同理：

```python
Y_norms = np.sqrt(
    np.sum(Y ** 2, axis=1)
)
```

得到：

```text
[M]
```

---

## 9.4 用 Broadcasting 构造分母

现在：

```text
X_norms.shape = [N]

Y_norms.shape = [M]
```

我们需要得到：

```text
[N, M]
```

其中第 $(i,j)$ 个元素应该是：

$\|\mathbf{x}_i\|_2 \|\mathbf{y}_j\|_2$ 所以把：

```text
X_norms: [N]
```

变成：

```text
[N, 1]
```

代码：

```python
X_norms[:, None]
```

而：

```text
Y_norms: [M]
```

在 Broadcasting 时可以理解成：

```text
[1, M]
```

于是：

$[N,1] \odot [1,M] \rightarrow [N,M]$ 其中 $\odot$ 表示这里进行逐元素乘法并利用 Broadcasting。

代码：

```python
denominator = X_norms[:, None] * Y_norms
```

最终：

```python
similarities = dot_products / denominator
```

完整实现：

```python
dot_products = X @ Y.T

X_norms = np.sqrt(np.sum(X ** 2, axis=1))
Y_norms = np.sqrt(np.sum(Y ** 2, axis=1))

denominator = X_norms[:, None] * Y_norms

similarities = dot_products / denominator
```

---

# 10. Lab 2：Stable Softmax

Softmax 可以把一组 Logits 转换成概率。

对于一个向量：

$\operatorname{softmax}(\mathbf{x})_i = \frac{e^{x_i}} {\sum_{j=1}^{C}e^{x_j}}$ 其中 $C$ 是类别数量。

假设：

```text
logits.shape = [B, C]
```

其中：

* $B$：Batch Size
* $C$：Number of Classes

Softmax 要保证每一个样本自己的类别概率之和为 $1$：

$\sum_{j=1}^{C}p_{ij}=1$ 因此对于 `[B, C]`：

```python
sum(axis=1)
```

是在类别维度上进行 Reduction。

---

## 10.1 为什么不能直接 `exp(x)`？

假设：

```python
x = np.array([
    [1000., 1001., 1002.]
])
```

直接：

```python
np.exp(x)
```

需要计算：

$e^{1000},\qquad e^{1001},\qquad e^{1002}$ 这些数字极大，可能超出浮点数的表示范围，产生：

```text
inf
```

最终甚至可能出现：

$\frac{\infty}{\infty} \rightarrow \mathrm{NaN}$ 这就是 Numerical Instability（数值不稳定）。

---

# 11. Stable Softmax 为什么可以减最大值？

设某一行 Logits 为：

$x_1,x_2,\ldots,x_C$ Softmax：

$p_i = \frac{e^{x_i}} {\sum_j e^{x_j}}$ 给所有 Logits 同时减去常数 $c$：

$p_i = \frac{e^{x_i-c}} {\sum_j e^{x_j-c}}$ 利用：

$e^{x_i-c} = e^{x_i}e^{-c}$ 得到：

$p_i = \frac{e^{x_i}e^{-c}} {\sum_j e^{x_j}e^{-c}}$ 分子和分母中的 $e^{-c}$ 可以约掉：

$p_i = \frac{e^{x_i}} {\sum_j e^{x_j}}$ 因此：

> 给所有 Logits 同时减去相同常数，不会改变 Softmax 的结果。

为了数值稳定，通常选择：

$c=\max_j x_j$ 于是：

$\operatorname{softmax}(\mathbf{x})_i = \frac{ e^{x_i-\max(\mathbf{x})} }{ \sum_j e^{x_j-\max(\mathbf{x})} }$ 这样最大的指数输入变成 $0$：

$e^0=1$ 其他指数输入都小于或等于 $0$，从而显著降低 Overflow 风险。

---

## 11.1 NumPy 实现

```python
def stable_softmax(x):

    exp_x = np.exp(
        x - x.max(axis=1, keepdims=True)
    )

    sum_exp = exp_x.sum(
        axis=1,
        keepdims=True
    )

    return exp_x / sum_exp
```

Shape 流程：

```text
x
[B, C]

↓

row_max
[B, 1]

↓

x - row_max
[B, C]

↓

exp
[B, C]

↓

sum(axis=1, keepdims=True)
[B, 1]

↓

Broadcasting Division

↓

probs
[B, C]
```

最后可以验证：

```python
probs.sum(axis=1)
```

理论上：

$\sum_{j=1}^{C}p_{ij}=1$ 实际浮点计算中应该非常接近 $1$。

---

# 12. Lab 3：Cross Entropy

假设：

```text
logits.shape = [B, C]

labels.shape = [B]
```

例如：

```python
labels = np.array([0, 1, 3])
```

表示：

```text
sample 0 → 正确类别是 class 0

sample 1 → 正确类别是 class 1

sample 2 → 正确类别是 class 3
```

首先：

```python
probs = stable_softmax(logits)
```

得到：

```text
[B, C]
```

但计算 Cross Entropy 时，我们真正关心的是：

> 模型给每一个样本的正确类别分配了多少概率？

---

## 12.1 Advanced Indexing

假设：

```text
labels = [0, 1, 3]
```

我们需要：

```text
probs[0, 0]

probs[1, 1]

probs[2, 3]
```

首先：

```python
np.arange(3)
```

得到：

```text
[0, 1, 2]
```

所以：

```python
rows = np.arange(len(labels))
```

有：

```text
rows   = [0, 1, 2]

labels = [0, 1, 3]
```

NumPy 会将它们配对成：

```text
(0, 0)
(1, 1)
(2, 3)
```

因此：

```python
correct_probs = probs[
    np.arange(len(labels)),
    labels
]
```

得到：

```text
correct_probs.shape = [B]
```

---

# 13. Cross Entropy Loss

对于一个样本，如果正确类别的预测概率是 $p_y$，Cross Entropy Loss 为：

$L=-\log p_y$ 对于一个 Batch：

$L = -\frac{1}{B} \sum_{i=1}^{B} \log p_{i,y_i}$ 其中：

* $B$：Batch Size
* $y_i$：第 $i$ 个样本的正确类别
* $p_{i,y_i}$：模型对第 $i$ 个样本正确类别分配的概率

NumPy：

```python
losses = -np.log(correct_probs)

loss = np.mean(losses)
```

Shape 流程：

```text
logits
[B, C]

↓

Stable Softmax

↓

probs
[B, C]

↓

Advanced Indexing

↓

correct_probs
[B]

↓

-log

↓

losses
[B]

↓

mean

↓

loss
scalar
```

---

## 13.1 Cross Entropy 的直觉

如果模型对正确答案非常有信心：

$p_y=0.9$ 那么：

$-\log(0.9)\approx0.105$ Loss 很小。

如果模型给正确答案的概率很低：

$p_y=0.1$ 那么：

$-\log(0.1)\approx2.303$ Loss 很大。

因此：

> **模型越相信正确答案，Cross Entropy Loss 越小。**

---

# 14. Lab 4：Mini-Batch 神经网络运算

假设：

```text
X: [B, D]

W: [D, H]

b: [H]
```

神经网络中非常经典的一步是：

$Z=XW+b$ 其中：

* $B$：Batch Size
* $D$：Input Dimension
* $H$：Hidden Dimension

NumPy：

```python
Z = X @ W + b
```

首先：

$[B,D]\times[D,H] \rightarrow [B,H]$ 然后：

```text
Z: [B, H]

b:    [H]
```

Broadcasting 时：

```text
[H]

↓

[1, H]

↓

[B, H]
```

所以：

$[B,H]+[H] \rightarrow [B,H]$---

# 15. ReLU

ReLU（Rectified Linear Unit）定义为：

$\operatorname{ReLU}(x) = \max(0,x)$ 也就是：

$\operatorname{ReLU}(x) = \begin{cases} x, & x>0\\ 0, & x\leq0 \end{cases}$ NumPy：

```python
H = np.maximum(Z, 0)
```

这是逐元素运算，因此不会改变 Shape：

```text
Z: [B, H]

↓

ReLU

↓

H: [B, H]
```

整个小型神经网络流程：

$X \rightarrow XW+b \rightarrow Z \rightarrow \operatorname{ReLU}(Z) \rightarrow H$ 对应 Shape：

$[B,D] \rightarrow [B,H] \rightarrow [B,H]$---

# 16. Lab 5：SVD 与二维可视化

假设：

```text
X.shape = [N, D]
```

其中：

* $N$：数据点数量
* $D$：每个数据点的特征维度

我们的目标是：

$[N,D] \rightarrow [N,2]$ 这样每一个数据点最终只有：

$(x_i,y_i)$ 两个坐标，就可以画在二维平面上。

---

# 17. 数据中心化

首先计算每一个 Feature 在所有样本上的平均值：

```python
mean = X.mean(
    axis=0,
    keepdims=True
)
```

Shape：

```text
X:    [N, D]

mean: [1, D]
```

数学上，第 $j$ 个 Feature 的均值：

$\mu_j = \frac{1}{N} \sum_{i=1}^{N}X_{ij}$ 中心化：

$X_{ij}^{\text{centered}} = X_{ij}-\mu_j$ 矩阵形式：

$X_{\text{centered}} = X-\mu$ NumPy：

```python
X_centered = X - mean
```

利用 Broadcasting：

```text
[N, D]

-

[1, D]

↓

[N, D]
```

---

# 18. SVD

执行：

```python
U, S, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)
```

SVD 将矩阵分解为：

$X_{\text{centered}} = U\Sigma V^{\mathsf T}$ 其中：

* $U$：左奇异向量
* $\Sigma$：奇异值组成的对角矩阵
* $V^{\mathsf T}$：右奇异向量的转置

例如：

```text
X_centered.shape = (6, 3)
```

使用：

```python
full_matrices=False
```

得到：

```text
U  → (6, 3)

S  → (3,)

Vt → (3, 3)
```

NumPy 返回的 `S` 是奇异值的一维数组，而不是直接返回完整的 $\Sigma$ 对角矩阵。

Day 1 暂时不需要证明 SVD。

目前最重要的直觉是：

> **SVD 可以帮助我们找到高维数据中最重要的方向。**

---

# 19. 从 3D 降到 2D

取前两个主要方向：

```python
Vt[:2]
```

Shape：

```text
(2, 3)
```

转置：

```python
Vt[:2].T
```

得到：

```text
(3, 2)
```

然后进行投影：

```python
X_2d = X_centered @ Vt[:2].T
```

Shape：

$(6,3)\times(3,2) \rightarrow (6,2)$ 更一般地：

$[N,D]\times[D,2] \rightarrow [N,2]$ 也就是说：

```text
原来：

N 个数据点
每个点 D 个特征

↓

现在：

N 个数据点
每个点 2 个特征
```

数据点数量 $N$ 没有变化。

变化的是每个数据点的表示维度：

$D\rightarrow2$---

# 20. Scatter Plot

现在：

```text
X_2d.shape = (6, 2)
```

其中：

```python
X_2d[:, 0]
```

表示所有数据点的第 $0$ 个坐标，即 x 坐标。

而：

```python
X_2d[:, 1]
```

表示所有数据点的第 $1$ 个坐标，即 y 坐标。

所以：

```python
import matplotlib.pyplot as plt

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1]
)

plt.xlabel("SVD Dimension 1")
plt.ylabel("SVD Dimension 2")
plt.title("2D Projection using SVD")

plt.show()
```

以后 Word Embedding 可能是：

$\mathbf{e}\in\mathbb{R}^{300}$ 甚至：

$\mathbf{e}\in\mathbb{R}^{768}$ 或者更高维。

人类无法直接观察高维空间，所以可以：

```text
高维 Embedding

↓

降维

↓

二维表示

↓

Scatter Plot
```

从而观察不同向量之间的结构关系。

---

# 21. Vectorization（向量化）

今天另一个非常重要的概念：

> **尽量使用矩阵 / 数组运算表达计算，而不是 Python for-loop。**

例如 Pairwise Cosine Similarity。

最直接的方法可能是：

```python
for x in X:
    for y in Y:
        # cosine(x, y)
        ...
```

但是 NumPy 可以通过：

```python
X @ Y.T
```

一次计算所有 Pairwise Dot Products：

$D_{ij} = \mathbf{x}_i^{\mathsf T}\mathbf{y}_j$ 从而得到：

$D\in\mathbb{R}^{N\times M}$ Vectorization 不仅仅是让代码更短。

更重要的是：

> **这种表达方式更接近数学本身，也更接近以后 PyTorch / GPU 上的 Tensor 运算。**

---

# 22. 今天值得记住的 Shape 模式

## 22.1 Linear Layer

$[B,D]\times[D,H] \rightarrow [B,H]$---

## 22.2 Bias Broadcasting

$[B,H]+[H] \rightarrow [B,H]$---

## 22.3 Pairwise Similarity

$[N,D]\times[D,M] \rightarrow [N,M]$---

## 22.4 Feature Reduction

原始 Shape：

```text
[B, D]
```

沿 Feature Dimension：

```python
axis=1
```

Reduction 后：

```text
[B]
```

如果：

```python
keepdims=True
```

则：

```text
[B, 1]
```

---

## 22.5 Transformer 中常见的 Broadcasting

$[B,T,D]+[D]$ 右对齐后可以理解成：

$[B,T,D]+[1,1,D]$ 最终：

$[B,T,D]$---

## 22.6 降维

$[N,D]\times[D,2] \rightarrow [N,2]$---

# 23. 今天真实踩过的坑

## 坑 1：`np.nums` 写错

我写过：

```python
np.nums(...)
```

正确应该是：

```python
np.sum(...)
```

如果以后看到：

```text
module 'numpy' has no attribute ...
```

首先检查：

> API 名字是不是拼错了？

---

## 坑 2：混淆 `(2,1)` 和 `(1,2)`

Cosine Similarity 中：

```text
X_norms: (2,)

Y_norms: (4,)
```

我一开始以为应该把 $X$ 变成：

```text
(1, 2)
```

但我们真正需要的 Pairwise 结果是：

```text
             Y₀    Y₁    Y₂    Y₃

X₀            ·     ·     ·     ·

X₁            ·     ·     ·     ·
```

所以：

```text
X 控制行
Y 控制列
```

应该：

```text
X_norms → (2, 1)

Y_norms → (1, 4)
```

然后：

$(2,1)\odot(1,4) \rightarrow (2,4)$---

## 坑 3：把补 `1` 和 Broadcast 后的大小混淆

例如：

```text
A: (5, 7, 10)

b:       (10,)
```

不是直接把：

```text
(10,)
```

补成：

```text
(5, 7, 10)
```

而是先：

```text
(10,)

↓

(1, 1, 10)
```

然后：

```text
(1, 1, 10)

↓

(5, 7, 10)
```

所以：

> **补维度得到的是 $1$，Broadcasting 才负责把 $1$ 扩展。**

---

## 坑 4：混淆 `[:, 0]` 与 `[:, None]`

我在 SVD Scatter Plot 的时候写过类似：

```python
X_2d[:, None]
```

但真正需要的是：

```python
X_2d[:, 0]
X_2d[:, 1]
```

因为：

```text
[:, 0]

→ 取第 0 列
```

而：

```text
[:, None]

→ 新增一个维度
```

---

# 24. 以后看到 Tensor 运算时的检查清单

以后遇到 NumPy / PyTorch 代码，先不要急着运行。

按照下面的顺序检查。

### Step 1：写 Shape

例如：

```text
X: [B, T, D]
```

### Step 2：给每个维度语义

不要只看到：

```text
32 × 128 × 768
```

而应该想到：

```text
Batch × Token × Feature
```

### Step 3：如果有 Reduction

问自己：

> 哪个 Axis 被消掉？

例如：

```python
X.mean(axis=1)
```

如果：

```text
X: [B, T, D]
```

那么：

```text
axis=1

↓

T 消失

↓

[B, D]
```

### Step 4：需不需要 `keepdims=True`？

如果后面还要与原 Tensor 做 Broadcasting，保留大小为 $1$ 的维度通常很有帮助。

### Step 5：如果发生 Broadcasting

永远：

> **从右往左比较。**

缺少的维度：

> **在左边视为 $1$。**

### Step 6：如果是矩阵乘法

检查：

$[A,B]\times[B,C]$ 中间维度是否相同。

输出：

$[A,C]$

### Step 7：运行代码之前预测 Shape

这是今天最重要的习惯之一。

---

# 25. 与 Deep Learning / CS224N 的联系

今天看起来只是在学习 NumPy，但实际上已经接触到了很多神经网络的核心运算。

---

## 25.1 Embedding Similarity

```python
X @ Y.T
```

可以一次计算大量向量之间的 Dot Product。

配合 Norm 后：

$\operatorname{cosine}(\mathbf{x},\mathbf{y}) = \frac{ \mathbf{x}^{\mathsf T}\mathbf{y} }{ \|\mathbf{x}\|_2\|\mathbf{y}\|_2 }$ 以后 Word Vector / Embedding 会大量使用类似思想。

---

## 25.2 Neural Network

今天已经实现：

$Z=XW+b$ 以及：

$H=\operatorname{ReLU}(Z)$ 组合起来：

$H = \operatorname{ReLU}(XW+b)$ 这已经是最基础的神经网络层结构之一。

---

## 25.3 Classification

今天已经走过：

```text
Logits

↓

Stable Softmax

↓

Probabilities

↓

取正确类别概率

↓

-log

↓

Mean

↓

Loss
```

对应数学形式：

$p_{i,c} = \frac{ e^{z_{i,c}} }{ \sum_{j=1}^{C}e^{z_{i,j}} }$ 以及：

$L = -\frac{1}{B} \sum_{i=1}^{B} \log p_{i,y_i}$ 所以以后 PyTorch 中看到：

```python
F.cross_entropy(logits, labels)
```

就不会完全把它当成一个黑盒。

---

## 25.4 Attention

以后 Transformer Attention 中会出现：

$QK^{\mathsf T}$ 然后：

$\operatorname{softmax} \left( \frac{QK^{\mathsf T}}{\sqrt{d_k}} \right)$ 再与 $V$ 做矩阵乘法：

$\operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^{\mathsf T}}{\sqrt{d_k}} \right)V$ 所以今天学习的：

```text
Shape
Matrix Multiplication
Axis
Broadcasting
Softmax
```

都是后面理解 Attention 的基础。

现在暂时不需要理解完整 Attention 公式，只需要知道：

> **今天练习的 Tensor Thinking 会直接用到 Transformer。**

---

## 25.5 Word Embeddings

今天 SVD 做的是：

$\mathbb{R}^{D} \rightarrow \mathbb{R}^{2}$ 也就是：

```text
高维向量

↓

低维表示

↓

二维可视化
```

后面学习 Word Vectors 时会继续遇到这个思想。

---

# 26. Day 1 最重要的四条规则

如果过几天忘了今天的大部分内容，至少记住下面四条。

### 1. Shape 中的每一个位置对应一个 Axis

例如：

```text
[B, T, D]
 ↑  ↑  ↑
 0  1  2
```

### 2. `reduction(axis=n)` 会压缩第 n 个维度

例如：

$[B,T,D] \xrightarrow{\text{mean(axis=1)}} [B,D]$

### 3. `keepdims=True` 会把被压缩的维度保留成大小为 1

例如：

$[B,D] \xrightarrow{\text{mean(axis=1, keepdims=True)}} [B,1]$

### 4. Broadcasting 从最右边开始比较

兼容条件：

```text
相同              ✓

其中一个是 1      ✓

缺少维度          ✓

否则              ✗
```

例如：

$[B,T,D]+[D]$ 可以理解成：

$[B,T,D]+[1,1,D]$ 最终：

$[B,T,D]$---

# 27. Day 1 总结

今天真正学到的不是：

> “我会 NumPy 了。”

而应该是：

> **我开始能够通过 Shape 推理 Tensor 运算。**

今天逐渐形成的思考流程：

```text
这个 Tensor 表示什么？

↓

每个维度的语义是什么？

↓

Shape 是什么？

↓

操作发生在哪个 Axis？

↓

Shape 会怎么变化？

↓

是否需要 Broadcasting？

↓

矩阵乘法是否兼容？

↓

先预测结果

↓

再运行代码验证
```

对于之后的 CS224N，我希望继续保持：

> **先看语义 → 再看 Shape → 再看运算 → 最后运行代码。**

---

# 28. Day 1 完成情况

* [X]  NumPy Array 基础
* [X]  Shape / ndim
* [X]  Indexing / Slicing
* [X]  Axis
* [X]  Reduction
* [X]  `keepdims`
* [X]  Broadcasting
* [X]  Matrix Multiplication
* [X]  Vectorization
* [X]  Pairwise Cosine Similarity
* [X]  Stable Softmax
* [X]  Cross Entropy
* [X]  Mini-batch Matrix Operations
* [X]  Bias Broadcasting
* [X]  ReLU
* [X]  SVD
* [X]  2D Visualization

---

# 29. 下一步

**Day 2 — PyTorch Fundamentals**

下一步的核心目标：

> **把今天建立的 NumPy Shape-first Thinking 转移到 PyTorch Tensor，并开始理解 Autograd 与神经网络训练。**
>
