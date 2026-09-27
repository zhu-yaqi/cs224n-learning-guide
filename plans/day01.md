# CS224N Day 1 --- NumPy for Deep Learning

> **今日目标**：不是"学完 NumPy"，而是建立后续 PyTorch / Transformer
> 最需要的能力：\
> **看到一个数组/矩阵操作，能判断输入 shape → 操作 axis → broadcasting →
> 输出 shape。**
>
> **预计时间**：约 5.5--6 小时\
> **今日公开产物**： - `notes/00_numpy_for_dl.md` -
> `labs/numpy_bootcamp/numpy_bootcamp.ipynb`
>
> **核心原则**：看教程不是进度；能自己推
> shape、写代码、解释错误并完成复盘才算进度。

------------------------------------------------------------------------

## 今日完成标准

今天结束前，检查自己是否能做到：

-   [ ] 看到常见矩阵运算，能在运行前判断主要 shape
-   [ ] 能解释 `axis=0` 和 `axis=1`
-   [ ] 能解释 NumPy broadcasting 为什么成立 / 为什么报错
-   [ ] 能不用 Python 双重 `for` loop 实现 pairwise cosine similarity
-   [ ] 能自己实现 stable softmax
-   [ ] 能自己实现基础 cross entropy
-   [ ] 完成 `numpy_bootcamp.ipynb` 的 5 个 mini-lab
-   [ ] 完成 `notes/00_numpy_for_dl.md`
-   [ ] 完成 Day 1 Git commit
-   [ ] 记录至少 3 个未来可以做成自媒体内容的问题/发现

------------------------------------------------------------------------

# 1. 先建立今天的工作区｜20 min

## GitHub / 本地目录

公开仓库建议保持：

``` text
cs224n-learning-guide/
├── README.md
├── notes/
│   └── 00_numpy_for_dl.md
├── labs/
│   └── numpy_bootcamp/
│       └── numpy_bootcamp.ipynb
├── assignment-reflections/
├── project/
├── content-ideas.md
└── resources.md
```

今天只需要重点维护：

``` text
notes/00_numpy_for_dl.md
labs/numpy_bootcamp/numpy_bootcamp.ipynb
content-ideas.md
```

### GitHub 原则

**Public repo 可以放：** - 自己写的学习笔记 - 自己设计的 toy
implementation - mini-lab - bug / failure analysis - 自己画的图 -
自己的实验和项目

**不要公开：** - Stanford assignment 完整 TODO 答案 - 可直接提交
Gradescope 的完整作业代码 - written assignment 的逐题答案

正式课程作业以后单独放 private repo。

------------------------------------------------------------------------

# 2. NumPy 最低必要基础｜约 90 min

## 学习资源

优先：

1.  CS231N Python / NumPy Tutorial\
    https://cs231n.github.io/python-numpy-tutorial/

2.  NumPy Quickstart\
    https://numpy.org/doc/stable/user/quickstart

3.  CS224N Python Review slides / Colab（从当年课程页面进入）

> 不需要把 NumPy 文档从头读到尾。

## 今天必须掌握

### Array 基础

-   [ ] `np.array`
-   [ ] `.shape`
-   [ ] `.ndim`
-   [ ] `.dtype`

### Shape 操作

-   [ ] `reshape`
-   [ ] `transpose`
-   [ ] `squeeze`

### Indexing

-   [ ] slicing
-   [ ] integer indexing
-   [ ] boolean indexing

### Reduction

-   [ ] `sum`
-   [ ] `mean`
-   [ ] `max`
-   [ ] `argmax`
-   [ ] `axis`
-   [ ] `keepdims`

### 核心计算

-   [ ] broadcasting
-   [ ] `@`
-   [ ] `dot`
-   [ ] `stack`
-   [ ] `concatenate`
-   [ ] `np.linalg.svd`
-   [ ] vectorization

------------------------------------------------------------------------

## 学习规则：先猜 shape，再运行

以后看到：

``` python
X.shape == (32, 128)
W.shape == (128, 64)

Y = X @ W
```

不要立刻运行。

先写：

``` text
(32, 128) @ (128, 64)
            ↓
         (32, 64)
```

然后再运行验证。

### 今日习惯

> **每看到一个 NumPy expression，都问自己：输入 shape
> 是什么？哪个维度发生了变化？输出 shape 是什么？**

------------------------------------------------------------------------

# 3. 专攻 Axis + Broadcasting｜约 40 min

这是今天最值得慢下来理解的部分。

不运行代码，先在纸上 / Markdown 中回答：

``` python
X.shape = (32, 128)

X.sum(axis=0).shape
# ?

X.sum(axis=1).shape
# ?

X.mean(axis=0, keepdims=True).shape
# ?
```

然后：

``` python
X.shape = (32, 128)

b1.shape = (128,)
X + b1
# 合法吗？结果 shape 是什么？为什么？

b2.shape = (32,)
X + b2
# 合法吗？为什么？
```

## 自测

-   [ ] 我能用自己的话解释 `axis=0`
-   [ ] 我能用自己的话解释 `axis=1`
-   [ ] 我知道 reduction 后为什么某个维度会消失
-   [ ] 我知道 `keepdims=True` 有什么用
-   [ ] 我能从右向左检查 broadcasting compatibility

### 一句话复盘

写下：

> **我现在对 axis 的理解：**

...

> **我现在对 broadcasting 的理解：**

...

------------------------------------------------------------------------

# 4. NumPy Bootcamp｜约 2 h

打开已有文件：

``` text
labs/numpy_bootcamp/numpy_bootcamp.ipynb
```

不要重新创建 notebook，在现有 notebook 上继续。

------------------------------------------------------------------------

## Lab 1 --- Pairwise Cosine Similarity

假设：

``` text
X: [N, D]
Y: [M, D]
```

目标：

``` text
similarity: [N, M]
```

要求：

-   [ ] 不使用 Python 双重 `for` loop
-   [ ] 写出 cosine similarity 公式
-   [ ] 标注每一步 shape
-   [ ] 解释为什么矩阵乘法可以得到所有 pair 的 dot product
-   [ ] 用一个很小的输入手动 sanity check

重点理解：

``` text
[N, D] @ [D, M] -> [N, M]
```

### 我遇到的问题

...

------------------------------------------------------------------------

## Lab 2 --- Stable Softmax

自己实现 softmax，不直接调用现成 softmax。

要求：

-   [ ] 实现普通版本
-   [ ] 构造一个大数输入观察数值问题
-   [ ] 实现 stable softmax
-   [ ] 支持 batch 输入
-   [ ] 明确 softmax 所在的 axis

回答：

### 为什么直接 `exp(x)` 可能出问题？

...

### 为什么减去 `max(x)` 不改变 softmax 结果？

...

### logits `[B, C]` 时，softmax 应该在哪个 axis？

...

------------------------------------------------------------------------

## Lab 3 --- Cross Entropy

假设：

``` text
logits: [B, C]
labels: [B]
loss: scalar
```

要求：

-   [ ] 从 logits / probabilities 推到 loss
-   [ ] 不直接调用现成 cross entropy
-   [ ] 正确取得每个样本对应 target class 的 probability / log
    probability
-   [ ] batch 后得到 scalar loss
-   [ ] 标注关键变量 shape

回答：

> 为什么 labels 是 `[B]`，而 logits 是 `[B, C]`？

...

------------------------------------------------------------------------

## Lab 4 --- Mini-batch Matrix Operations

模拟一个最简单的神经网络线性层：

``` text
X: [B, D]
W: [D, H]
b: [H]
```

实现：

``` python
H = X @ W + b
```

要求：

-   [ ] 判断 `X @ W` shape
-   [ ] 判断最终 `H` shape
-   [ ] 解释 `b` 如何 broadcast
-   [ ] 加一个简单 activation
-   [ ] 尝试故意制造一个 shape mismatch，然后解释报错

### Shape table

  Tensor    Meaning             Shape
  --------- ------------------- ----------
  `X`       batch input         `[B, D]`
  `W`       weight              `[D, H]`
  `b`       bias                `[H]`
  `X @ W`   linear output       `[B, H]`
  `H`       output after bias   `[B, H]`

------------------------------------------------------------------------

## Lab 5 --- SVD → 2D Visualization

目标：建立"高维向量 → 低维表示"的直觉，为后面的 Word Vectors 做准备。

要求：

-   [ ] 创建一个小型数据矩阵
-   [ ] 使用 `np.linalg.svd`
-   [ ] 检查返回矩阵的 shape
-   [ ] 使用两个维度得到 2D representation
-   [ ] 用 scatter plot 可视化
-   [ ] 用自己的话写一句 SVD 在这里做了什么

> 今天不要求深入证明 SVD。

### 我的理解

...

------------------------------------------------------------------------

# 5. 闭卷写技术笔记｜约 45 min

关闭：

-   教程
-   文档
-   Notebook
-   ChatGPT

打开：

``` text
notes/00_numpy_for_dl.md
```

先凭记忆写，写完再查资料补缺。

建议结构：

``` markdown
# NumPy for Deep Learning

## 1. Problem
为什么深度学习需要熟练的 NumPy / tensor 操作？

## 2. Core Ideas
- shape
- axis
- broadcasting
- vectorization

## 3. Math
- matrix multiplication
- cosine similarity
- softmax
- cross entropy
- SVD

## 4. Shapes
记录今天最重要的 shape flow。

## 5. Minimal Implementations
放自己的 toy implementation / pseudocode。

## 6. Failure Modes / Common Bugs
记录今天真正踩过的坑。

## 7. Connection to PyTorch / Transformer / CS336
今天的内容以后在哪里会再次出现？
```

## 必须有一个 Shape Section

例如：

``` text
X: [B, D]
W: [D, H]
b: [H]

X @ W       -> [B, H]
X @ W + b   -> [B, H]
```

## 必须有一个 Bugs Section

不要写成 API 手册。

记录真正发生过的错误：

``` markdown
## Bugs I hit today

### 1. Confused axis=0 and axis=1
What I thought:
...

What actually happened:
...

How I will reason about it next time:
...
```

今天至少记录：

-   [ ] 1 个 shape / axis 问题
-   [ ] 1 个 broadcasting 问题或观察
-   [ ] 1 个 numerical stability / vectorization 问题或观察

------------------------------------------------------------------------

# 6. GitHub 收尾｜约 30 min

## 今天公开的内容

``` text
notes/00_numpy_for_dl.md
labs/numpy_bootcamp/numpy_bootcamp.ipynb
```

检查：

-   [ ] Notebook 从头到尾可以运行
-   [ ] 删除无意义的临时输出
-   [ ] Markdown 能解释实验目的
-   [ ] 没有复制课程作业答案
-   [ ] 没有 API key / token / private data
-   [ ] Notes 是自己的理解，不是教程复制粘贴

建议 commit：

``` bash
git add notes/00_numpy_for_dl.md labs/numpy_bootcamp/numpy_bootcamp.ipynb
git commit -m "day01: numpy fundamentals and vectorization bootcamp"
```

------------------------------------------------------------------------

# 7. 自媒体素材池｜10--15 min

> **今天只收集素材，不做视频，不做精致排版。**

打开：

``` text
content-ideas.md
```

记录今天真正让你产生"原来如此"的东西。

可以从这些方向开始：

``` markdown
## Day 1

- 为什么学 Transformer 前一定要搞懂 tensor shape？
- 我终于搞懂 NumPy axis=0 / axis=1 了
- Broadcasting 到底在干嘛？
- 为什么 softmax 要先减 max？
- 不用 for-loop 怎么一次计算所有 embedding 的 cosine similarity？
- 从零系统学 CS224N：Day 1 我真正需要补的 NumPy
```

但优先记录**你今天真实遇到的问题**。

### 今日素材

1.  ...
2.  ...
3.  ...

------------------------------------------------------------------------

# 8. Day 1 最终自测｜15 min

不要看笔记，回答下面的问题。

### Q1

``` text
X: [32, 128]
W: [128, 64]

X @ W -> ?
```

答案：\_\_\_\_\_\_\_\_

### Q2

``` text
X: [32, 128]

X.sum(axis=0) -> ?
X.sum(axis=1) -> ?
```

答案：

``` text
axis=0:
axis=1:
```

### Q3

``` text
X: [32, 128]
b: [128]

X + b -> ?
```

为什么？

...

### Q4

如果：

``` text
X: [N, D]
Y: [M, D]
```

如何在不写双重 for-loop 的情况下得到：

``` text
pairwise cosine similarity: [N, M]
```

思路：

...

### Q5

为什么 stable softmax 通常先做：

``` python
x = x - max(x)
```

...

### Q6

``` text
logits: [B, C]
labels: [B]
```

为什么这两个 tensor 的 shape 不一样？

...

### Q7

今天学到的 NumPy shape / broadcasting 知识，为什么会影响以后理解
Transformer？

...

------------------------------------------------------------------------

# 9. 今天结束前的 5 分钟复盘

## 今天最重要的三个收获

1.  ...
2.  ...
3.  ...

## 今天最卡的地方

...

## 我是怎么解决的

...

## 明天开始 PyTorch 前，我还有什么没搞懂？

...

## 今天值得以后写成内容的一个主题

...

------------------------------------------------------------------------

# Definition of Done

只有下面这些完成，Day 1 才算真正结束：

``` text
[ ] NumPy 核心操作理解
[ ] axis / broadcasting 自测通过
[ ] 5 个 numpy_bootcamp mini-labs
[ ] 00_numpy_for_dl.md
[ ] 至少记录 3 个真实学习问题 / bug / insight
[ ] Git commit
[ ] content-ideas.md 有 Day 1 素材
```

**不要为了"打勾"赶进度。**

如果某个核心概念------尤其是
`shape / axis / broadcasting / vectorization`------还解释不清楚，就停下来把它搞懂。

------------------------------------------------------------------------

## Day 1 → Day 2

完成今天后，Day 2 进入：

**PyTorch 必备能力**

你会把今天的：

``` text
NumPy array
shape
broadcasting
softmax
cross entropy
matrix operations
```

迁移到：

``` text
torch.Tensor
autograd
nn.Module
loss.backward()
optimizer.step()
training loop
```

所以今天真正要带走的不是一堆 NumPy API，而是：

> **shape-first thinking + vectorized thinking。**
