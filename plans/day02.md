# CS224N Day 2 --- PyTorch Core：从 NumPy 到 Autograd 与 Training Loop

> **今日目标**：不是"把 PyTorch 教程看完"，而是把 Day 1 建立的\
> **shape-first thinking + vectorized thinking** 迁移到
> `torch.Tensor`，并真正理解一个神经网络是如何完成：
>
> **forward → loss → backward → update**
>
> **预计时间**：约 6 小时\
> **今日公开产物**：
>
> -   `notes/01_pytorch_core.md`
> -   `labs/autograd_sanity_checks.ipynb`
>
> **核心原则**：PyTorch API 不是重点。今天真正要带走的是：
>
> 1.  Tensor 和 NumPy ndarray 有什么相同与不同；
> 2.  PyTorch 如何自动记录计算并求梯度；
> 3.  `nn.Module` 如何组织模型和参数；
> 4.  一个最小 Training Loop 为什么必须有
>     `zero_grad → forward → loss → backward → step`。

------------------------------------------------------------------------

## 今日完成标准

今天结束前，检查自己是否能做到：

-   [ ] 能解释 `torch.Tensor` 与 NumPy `ndarray` 的主要区别
-   [ ] 能创建 Tensor，并判断 `shape / dtype / device`
-   [ ] 能在 NumPy 与 Tensor 之间转换
-   [ ] 能解释 CPU / GPU device 的基本概念
-   [ ] 能解释 `requires_grad=True` 是干什么的
-   [ ] 能解释 `.backward()` 做了什么
-   [ ] 能读取参数的 `.grad`
-   [ ] 知道为什么梯度需要清零
-   [ ] 能用 PyTorch 重写 Day 1 的 stable softmax
-   [ ] 能用 PyTorch 重写 Day 1 的 cross entropy
-   [ ] 能手算一个简单函数的梯度，并用 Autograd 验证
-   [ ] 能自己定义一个最小 `nn.Module`
-   [ ] 能找到模型的 parameters
-   [ ] 能解释 `model.train()` / `model.eval()`
-   [ ] 能解释 `torch.no_grad()`
-   [ ] 能从零写出最小 Training Loop
-   [ ] 完成 `labs/autograd_sanity_checks.ipynb`
-   [ ] 完成 `notes/01_pytorch_core.md`
-   [ ] 完成 Day 2 Git commit
-   [ ] 记录至少 3 个真实问题 / bug / 内容素材

------------------------------------------------------------------------

# 0. 今天开始前｜10 min

先进入你的项目：

``` text
cs224n-learning-guide/
```

今天建议新增：

``` text
cs224n-learning-guide/
├── notes/
│   └── 01_pytorch_core.md
│
├── labs/
│   └── autograd_sanity_checks.ipynb
│
└── content-ideas.md
```

先不要写完整笔记，只创建文件即可。

终端检查：

``` bash
git status
```

确认 Day 1 已经提交，工作区没有奇怪的未提交文件。

------------------------------------------------------------------------

# 1. 环境检查 + Tensor 基础｜约 55 min

## 1.1 学习资源

### 资源 A：PyTorch Learn the Basics

官方入口：

https://docs.pytorch.org/tutorials/beginner/basics/

今天**不是全部看完**。

第一阶段只看：

-   **Tensors**
-   可以快速扫一下 Quickstart，但不要陷进去

Tensors 直接链接：

https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

### 资源 B：CS224N PyTorch Tutorial

Stanford CS224N 课程主页：

https://web.stanford.edu/class/cs224n/

课程材料中的 PyTorch Tutorial：

https://web.stanford.edu/class/cs224n/materials/CS224N_PyTorch_Tutorial.html

今天它作为**辅助材料**，不要和 PyTorch 官方教程两边逐字重复学习。

------------------------------------------------------------------------

## 1.2 先检查 PyTorch

在今天的新 Notebook：

``` text
labs/autograd_sanity_checks.ipynb
```

第一格：

``` python
import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
```

如果 `torch.cuda.is_available()` 是 `False`，今天完全没关系。

> Day 2 的所有核心实验都可以在 CPU 上完成。

------------------------------------------------------------------------

## 1.3 从 Day 1 的 NumPy 映射到 PyTorch

你已经会：

``` python
import numpy as np

x = np.array([[1., 2.], [3., 4.]])
```

PyTorch：

``` python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
```

今天要观察：

``` python
x.shape
x.ndim
x.dtype
x.device
```

### 先建立映射

  NumPy          PyTorch
  -------------- ------------------
  `np.array`     `torch.tensor`
  `.shape`       `.shape`
  `.ndim`        `.ndim`
  `reshape`      `reshape`
  transpose      transpose / `.T`
  `@`            `@`
  `sum`          `sum`
  `mean`         `mean`
  broadcasting   broadcasting
  ndarray        Tensor

你的目标不是重新学一次数组，而是意识到：

> **Day 1 的 Shape / Axis / Broadcasting 思维基本可以直接迁移到
> Tensor。**

------------------------------------------------------------------------

## 1.4 今天必须掌握的 Tensor 操作

至少自己运行：

``` python
torch.tensor(...)
torch.zeros(...)
torch.ones(...)
torch.randn(...)

x.shape
x.ndim
x.dtype
x.device

x.reshape(...)
x.T

x.sum(dim=...)
x.mean(dim=...)

x @ W
```

注意 PyTorch 中经常使用：

``` python
dim=
```

而 NumPy 中你昨天主要使用：

``` python
axis=
```

例如：

``` python
x.sum(dim=1)
```

和昨天：

``` python
x.sum(axis=1)
```

在这里表达的是同一类维度操作。

------------------------------------------------------------------------

## 1.5 NumPy ↔ Tensor

练习：

``` python
a = np.array([1., 2., 3.])

t = torch.from_numpy(a)
```

再：

``` python
b = t.numpy()
```

至少回答：

> NumPy ndarray 和 PyTorch Tensor 的关系是什么？

> 为什么 PyTorch 还需要 Tensor，而不是一直使用 NumPy？

今天先记住最重要的两个原因：

``` text
Tensor
├── 可以运行在 GPU / accelerator 上
└── 可以参与 automatic differentiation
```

------------------------------------------------------------------------

## 1.6 CPU / GPU：今天只理解 Device

运行：

``` python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

device
```

然后：

``` python
x = torch.randn(2, 3)
x = x.to(device)

print(x.device)
```

今天只需要理解：

> **参与同一次运算的 Tensor / Model 通常需要在相同 device 上。**

不要今天开始研究 CUDA kernel、显存优化、混合精度。

这些都不是 Day 2 的任务。

------------------------------------------------------------------------

## 第一阶段自测

不看教程回答：

``` text
1. torch.Tensor 和 np.ndarray 最大的两个区别是什么？

2. NumPy 的 axis=1，在 PyTorch 中常写成什么？

3. Tensor 的 shape / ndim 思维和昨天有没有变化？

4. CPU Tensor 和 GPU Tensor 可以随便直接运算吗？
```

如果第 1--3 题能解释，就进入下一步。

------------------------------------------------------------------------

# 2. 把 Day 1 的 NumPy 改写成 PyTorch｜约 45 min

这一阶段非常重要。

**不要直接学新的数学。**

目标是证明：

> 我昨天学到的不是 NumPy API，而是 Tensor Thinking。

------------------------------------------------------------------------

## Task 1 --- Stable Softmax

昨天 NumPy 的核心逻辑是：

``` text
logits
→ subtract max
→ exp
→ sum
→ divide
```

今天用 PyTorch 重写。

假设：

``` python
logits = torch.tensor([
    [1000., 1001., 1002.],
    [2000., 1998., 2001.]
])
```

要求：

-   [ ] 不调用 `torch.softmax`
-   [ ] 找每一行 max
-   [ ] 保留维度
-   [ ] 减 max
-   [ ] `torch.exp`
-   [ ] 沿 class dimension 求和
-   [ ] 得到 probability
-   [ ] 检查每一行概率和约等于 1

Shape：

``` text
logits: [B, C]
row_max: [B, 1]
exp: [B, C]
sum_exp: [B, 1]
probs: [B, C]
```

### 思考

PyTorch 中：

``` python
torch.max(...)
```

的返回方式和 NumPy 的 `max` 不完全一样时，不要死猜。

先查看输出或文档，然后确认自己拿到的是：

``` text
values
```

还是：

``` text
values + indices
```

------------------------------------------------------------------------

## Task 2 --- Cross Entropy

继续使用：

``` text
logits: [B, C]
labels: [B]
```

要求：

-   [ ] 使用刚才自己的 stable softmax
-   [ ] 取得每个样本正确类别的 probability
-   [ ] `-log`
-   [ ] batch mean
-   [ ] 最终得到 scalar

不要直接调用：

``` python
torch.nn.functional.cross_entropy
```

完成以后，再用官方版本验证：

``` python
import torch.nn.functional as F

official_loss = F.cross_entropy(logits, labels)
```

比较：

``` text
你的 loss
vs
官方 loss
```

应该非常接近。

### 今天真正要理解

不是：

> "PyTorch 有 `F.cross_entropy`。"

而是：

> **以后即使调用一行 `F.cross_entropy`，我知道这行代码背后的 Shape
> 和数学流程。**

------------------------------------------------------------------------

# 3. Autograd：今天最重要的新知识｜约 75 min

## 3.1 学习资源

先看官方 Learn the Basics 中：

**Automatic Differentiation with `torch.autograd`**

https://docs.pytorch.org/tutorials/beginner/basics/autograd_tutorial.html

然后如果还不够清楚，再看：

**The Fundamentals of Autograd**

https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html

不要一开始就深挖 Autograd internals。

今天只需要搞懂：

``` text
requires_grad
computation graph
backward
.grad
gradient accumulation
no_grad
detach
```

------------------------------------------------------------------------

## 3.2 第一个 Autograd 实验

创建：

``` python
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2
```

先不要 `.backward()`。

自己手算：

$$
y = x^2
$$

所以：

$$
\frac{dy}{dx} = 2x
$$

当：

$$
x = 3
$$

应该：

$$
\frac{dy}{dx} = 6
$$

现在运行：

``` python
y.backward()
```

然后：

``` python
x.grad
```

检查是否为：

``` text
6
```

------------------------------------------------------------------------

## 3.3 第二个 Autograd 实验：Chain Rule

定义：

$$
y = (2x + 1)^2
$$

先手算：

令：

$$
u = 2x + 1
$$

则：

$$
y = u^2
$$

Chain Rule：

$$
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
$$

所以：

$$
\frac{dy}{dx}
=
2u \cdot 2
=
4(2x+1)
$$

选择一个简单的 $x$。

要求：

1.  先手算；
2.  再用 PyTorch；
3.  比较结果。

代码骨架：

``` python
x = torch.tensor(..., requires_grad=True)

y = ...

y.backward()

print(x.grad)
```

------------------------------------------------------------------------

## 3.4 第三个实验：多个参数

定义：

$$
y = wx + b
$$

然后定义一个简单 loss：

$$
L = (y - y_{\text{target}})^2
$$

令：

``` text
x = 常数
target = 常数
w = parameter
b = parameter
```

要求：

-   [ ] 手算 $\frac{\partial L}{\partial w}$
-   [ ] 手算 $\frac{\partial L}{\partial b}$
-   [ ] PyTorch Autograd 验证
-   [ ] 打印 `w.grad`
-   [ ] 打印 `b.grad`

这一题非常重要。

因为神经网络训练本质上就是：

> 对大量 parameters 自动计算 loss 的梯度。

------------------------------------------------------------------------

# 4. 梯度为什么要清零？｜约 25 min

这是 PyTorch 初学者非常容易踩的坑。

创建：

``` python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()

print(x.grad)
```

然后**不要清零**，再构造一次：

``` python
y = x ** 2
y.backward()

print(x.grad)
```

观察发生了什么。

你应该主动发现：

> PyTorch 默认会把梯度累积到 `.grad` 中。

所以训练模型时通常每轮需要：

``` python
optimizer.zero_grad()
```

或者在某些手写实验中：

``` python
x.grad.zero_()
```

------------------------------------------------------------------------

## 必须写进 Notebook 的一句话

> **`backward()` 计算梯度，但 PyTorch
> 默认累积梯度，因此正常训练循环中必须在合适的位置清零旧梯度。**

------------------------------------------------------------------------

# 5. `nn.Module`：模型到底是什么？｜约 60 min

## 5.1 学习资源

PyTorch 官方：

**Building the Neural Network**

https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

重点看：

-   `nn.Module`
-   `__init__`
-   `forward`
-   `nn.Linear`
-   parameters
-   model structure

不要今天深入 CNN。

教程里的视觉任务只是载体。

你真正需要的是模型组织方式。

------------------------------------------------------------------------

## 5.2 自己写最小模型

不要直接复制复杂网络。

自己写：

``` python
import torch
from torch import nn

class TinyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        # TODO

    def forward(self, x):
        # TODO
        return ...
```

目标结构：

``` text
X
↓
Linear
↓
ReLU
↓
Linear
↓
logits
```

Shape：

``` text
X: [B, D]

Linear 1:
[B, D] → [B, H]

ReLU:
[B, H] → [B, H]

Linear 2:
[B, H] → [B, C]

logits:
[B, C]
```

------------------------------------------------------------------------

## 5.3 检查 Parameters

创建模型后：

``` python
model = TinyClassifier(...)
```

观察：

``` python
for name, param in model.named_parameters():
    print(name, param.shape)
```

你应该看到类似：

``` text
layer1.weight
layer1.bias
layer2.weight
layer2.bias
```

然后回答：

> 为什么 ReLU 没有出现在 parameters 中？

> 为什么 Linear 会有 weight 和 bias？

> Optimizer 到底要更新谁？

------------------------------------------------------------------------

# 6. `train()` / `eval()` / `no_grad()`｜约 30 min

今天只建立正确概念。

------------------------------------------------------------------------

## 6.1 `model.train()`

表示：

> 模型进入训练模式。

某些层在训练和推理时行为不同，例如：

``` text
Dropout
BatchNorm
```

所以训练前通常：

``` python
model.train()
```

------------------------------------------------------------------------

## 6.2 `model.eval()`

表示：

> 模型进入 evaluation / inference 模式。

例如：

``` python
model.eval()
```

注意：

> `model.eval()` **不是**"关闭梯度"。

它改变的是某些模块的行为。

------------------------------------------------------------------------

## 6.3 `torch.no_grad()`

如果只是做 inference：

``` python
model.eval()

with torch.no_grad():
    predictions = model(x)
```

`torch.no_grad()` 表示：

> 这段运算不需要构建用于求梯度的计算图。

所以一定区分：

``` text
model.eval()
```

和：

``` text
torch.no_grad()
```

它们解决的不是同一个问题。

------------------------------------------------------------------------

# 7. 最小 Training Loop｜约 75 min

这是今天最终 Boss。

## 7.1 学习资源

PyTorch 官方：

**Optimizing Model Parameters**

https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

重点看：

``` text
loss function
optimizer
training loop
zero_grad
forward
loss
backward
step
```

不要被 FashionMNIST 的 Dataset 细节带跑。

今天重点不是 DataLoader。

------------------------------------------------------------------------

## 7.2 先记住 Training Loop 骨架

你最终必须能闭卷写出：

``` python
for epoch in range(num_epochs):

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()
```

今天最重要的是理解**每一行为什么存在**。

------------------------------------------------------------------------

## 7.3 五步分别在干什么？

### Step 1

``` python
optimizer.zero_grad()
```

清掉上一轮累积的梯度。

------------------------------------------------------------------------

### Step 2

``` python
logits = model(X)
```

Forward Pass。

从输入：

``` text
[B, D]
```

得到：

``` text
[B, C]
```

------------------------------------------------------------------------

### Step 3

``` python
loss = loss_fn(logits, y)
```

把模型预测和真实标签比较，得到一个 scalar loss。

``` text
logits: [B, C]
labels: [B]

↓

loss: scalar
```

------------------------------------------------------------------------

### Step 4

``` python
loss.backward()
```

Autograd 根据 Forward 时建立的计算关系，计算：

$$
\frac{\partial L}{\partial \theta}
$$

其中 $\theta$ 代表模型参数。

结果被存入参数的：

``` python
param.grad
```

------------------------------------------------------------------------

### Step 5

``` python
optimizer.step()
```

Optimizer 根据梯度更新参数。

最简单的 Gradient Descent 可以理解成：

$$
\theta
\leftarrow
\theta
-
\eta
\frac{\partial L}{\partial \theta}
$$

其中：

-   $\theta$：parameter
-   $\eta$：learning rate
-   $\frac{\partial L}{\partial \theta}$：gradient

------------------------------------------------------------------------

# 8. Mini-Lab：让一个小网络真的学起来｜约 35 min

在：

``` text
labs/autograd_sanity_checks.ipynb
```

最后做一个极小的分类任务。

不需要下载数据集。

自己造数据，例如：

``` text
X: [B, 2]
y: [B]
```

目标：

> 让 TinyClassifier 学会一个简单的二分类规律。

要求：

-   [ ] 创建 toy dataset
-   [ ] 建立 `TinyClassifier`
-   [ ] 使用 `nn.CrossEntropyLoss`
-   [ ] 使用 SGD 或 Adam
-   [ ] 自己写 Training Loop
-   [ ] 每隔若干 epoch 打印 loss
-   [ ] 确认 loss 有明显下降
-   [ ] 最后计算简单 accuracy

今天重点不是准确率有多高。

重点是：

``` text
数据
↓
模型
↓
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
参数更新
↓
Loss 下降
```

你要亲眼看到这条链真的工作。

------------------------------------------------------------------------

# 9. Autograd Sanity Checks Notebook 最终结构

你的：

``` text
labs/autograd_sanity_checks.ipynb
```

建议最后整理成：

``` text
# Day 2 — PyTorch & Autograd Sanity Checks

## 1. Tensor Basics
- shape
- dtype
- device
- NumPy ↔ Tensor

## 2. Stable Softmax in PyTorch

## 3. Cross Entropy from Scratch

## 4. Autograd Check 1
y = x²

## 5. Autograd Check 2
Chain Rule

## 6. Autograd Check 3
Linear Model Gradients

## 7. Gradient Accumulation Experiment

## 8. Tiny nn.Module

## 9. Minimal Training Loop

## 10. Toy Classification Experiment

## 11. Bugs / Observations
```

不要把 Notebook 变成教程复制粘贴。

每一个实验至少写一句：

> **我在验证什么？**

------------------------------------------------------------------------

# 10. 闭卷写技术笔记｜约 40 min

关闭：

-   PyTorch 官方教程
-   CS224N Tutorial
-   Notebook
-   ChatGPT

打开：

``` text
notes/01_pytorch_core.md
```

先凭记忆写。

建议结构：

``` markdown
# PyTorch Core for CS224N

## 1. Tensor vs NumPy

## 2. Shape / Dim / Broadcasting

## 3. Device: CPU vs GPU

## 4. Autograd
- requires_grad
- computation graph
- backward
- grad

## 5. Gradient Accumulation

## 6. nn.Module
- parameters
- forward

## 7. Loss

## 8. Optimizer

## 9. Minimal Training Loop

## 10. train / eval / no_grad

## 11. Bugs I Hit Today

## 12. Connection to CS224N / Transformer
```

------------------------------------------------------------------------

## 必须写清楚这个公式

神经网络训练的目标：

$$
\theta^\star
=
\arg\min_{\theta}
L(\theta)
$$

梯度下降最基本的更新：

$$
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L(\theta_t)
$$

然后用自己的话解释：

``` text
forward:
算预测

loss:
衡量预测有多差

backward:
算 loss 对 parameters 的梯度

optimizer.step:
根据梯度修改 parameters
```

------------------------------------------------------------------------

# 11. Day 2 最终自测｜约 20 min

不要看笔记。

------------------------------------------------------------------------

## Q1

NumPy：

``` python
x = np.zeros((32, 128))
```

PyTorch 对应怎么写？

答案：

...

------------------------------------------------------------------------

## Q2

``` text
X: [32, 128]
W: [128, 64]

X @ W -> ?
```

答案：

...

Day 1 的 Shape Thinking 到 PyTorch 后有没有变化？

...

------------------------------------------------------------------------

## Q3

下面是什么意思？

``` python
x = torch.tensor(
    3.0,
    requires_grad=True
)
```

`requires_grad=True` 到底要求 PyTorch 做什么？

...

------------------------------------------------------------------------

## Q4

如果：

$$
y=x^2
$$

且：

$$
x=4
$$

那么：

$$
\frac{dy}{dx}=?
$$

然后 PyTorch 中：

``` python
y.backward()
```

之后应该去哪里找这个梯度？

...

------------------------------------------------------------------------

## Q5

为什么训练循环中需要：

``` python
optimizer.zero_grad()
```

...

------------------------------------------------------------------------

## Q6

解释：

``` python
loss.backward()
```

和：

``` python
optimizer.step()
```

分别做什么。

...

------------------------------------------------------------------------

## Q7

`nn.Module` 中：

``` python
forward()
```

负责什么？

...

------------------------------------------------------------------------

## Q8

为什么：

``` python
model.eval()
```

不能完全替代：

``` python
with torch.no_grad():
```

...

------------------------------------------------------------------------

## Q9

把下面 Training Loop 的空填上：

``` python
for epoch in range(num_epochs):

    __________________

    logits = __________________

    loss = __________________

    __________________

    __________________
```

------------------------------------------------------------------------

## Q10

假设：

``` text
X: [B, D]
```

模型输出：

``` text
logits: [B, C]
```

分类标签为什么仍然通常是：

``` text
labels: [B]
```

而不是 `[B, C]`？

...

------------------------------------------------------------------------

# 12. 今天最重要的一张图

在笔记里自己画一遍：

``` text
              PARAMETERS
                  │
                  ▼
X [B,D] ────► MODEL ────► logits [B,C]
                              │
                              ▼
                         LOSS(logits, y)
                              │
                              ▼
                         scalar loss
                              │
                           backward
                              │
                              ▼
                     parameter.grad
                              │
                       optimizer.step
                              │
                              ▼
                    updated parameters
```

如果你能不看资料解释这张图，Day 2 的核心就已经掌握了。

------------------------------------------------------------------------

# 13. GitHub 收尾｜约 20 min

今天公开的主要产物：

``` text
notes/01_pytorch_core.md
labs/autograd_sanity_checks.ipynb
content-ideas.md
```

提交前检查：

-   [ ] Notebook 可以从头运行
-   [ ] 没有巨大无意义输出
-   [ ] 每个实验写了目的
-   [ ] Autograd 实验有"手算 vs PyTorch"对比
-   [ ] Training Loop 是自己写的
-   [ ] Notes 不是官方教程复制粘贴
-   [ ] 没有 API key / token / private data

先：

``` bash
git status
```

然后：

``` bash
git add notes/01_pytorch_core.md
git add labs/autograd_sanity_checks.ipynb
git add content-ideas.md
```

检查：

``` bash
git status
```

建议 commit：

``` bash
git commit -m "day02: pytorch autograd and training loop"
```

然后：

``` bash
git push
```

------------------------------------------------------------------------

# 14. 自媒体素材池｜10 min

今天仍然：

> **只收集，不制作。**

打开：

``` text
content-ideas.md
```

可以记录：

``` markdown
## Day 2

- NumPy 和 PyTorch Tensor 到底有什么区别？
- requires_grad=True 到底发生了什么？
- loss.backward() 到底做了什么？
- 为什么 PyTorch 每轮训练都要 zero_grad？
- model.eval() 和 torch.no_grad() 居然不是一回事
- 神经网络 Training Loop 其实只有 5 个核心步骤
- 我第一次手算梯度，再用 PyTorch Autograd 验证
```

优先记录今天你**真实卡住的问题**。

------------------------------------------------------------------------

# 15. 今天结束前 5 分钟复盘

## 今天最重要的三个收获

1.  ...
2.  ...
3.  ...

## 今天最卡的地方

...

## 我是怎么解决的

...

## 今天遇到的一个 PyTorch Bug

...

## 我现在能否闭卷写 Training Loop？

...

## 今天最值得以后做成内容的一个主题

...

------------------------------------------------------------------------

# Definition of Done

只有下面这些完成，Day 2 才算真正结束：

``` text
[ ] Tensor 基础掌握
[ ] NumPy → PyTorch 思维迁移完成
[ ] stable softmax PyTorch 版本
[ ] cross entropy PyTorch 手写版本
[ ] 至少 3 个 Autograd sanity checks
[ ] 理解 gradient accumulation
[ ] Tiny nn.Module
[ ] 最小 Training Loop
[ ] Toy model 的 loss 确实下降
[ ] autograd_sanity_checks.ipynb
[ ] 01_pytorch_core.md
[ ] content-ideas.md 有 Day 2 素材
[ ] Git commit + push
```

**不要为了完成 Day 2 去背 PyTorch API。**

真正的完成标准是：

> 给你一个小模型，你知道 Tensor 怎么流动、Loss 怎么产生、Gradient
> 怎么回来、Parameter 怎么更新。

------------------------------------------------------------------------

# Day 2 → Day 3

完成今天后，Day 3 进入：

**Word Vectors + Assignment 1**

你会开始把：

``` text
Tensor
Matrix Multiplication
Cosine Similarity
SVD
PyTorch / NumPy Shape Thinking
```

真正放进 NLP 的语境里：

``` text
word representation
embedding
similarity
co-occurrence
distributional semantics
```

Day 1 和 Day 2 是基础设施。

从 Day 3 开始，我们正式进入 **CS224N 的 NLP 主线**。
