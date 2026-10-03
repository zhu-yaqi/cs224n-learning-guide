# PyTorch Core — Tensor, Autograd, nn.Module & Training Loop

> CS224N Day 2  
> Goal: Understand the core PyTorch workflow behind neural network training.

---

## 1. PyTorch Tensor

PyTorch 的 `Tensor` 和 NumPy 的 `ndarray` 很像，都支持：

- 多维数组
- shape / dtype
- indexing
- broadcasting
- vectorized operations

```python
import torch

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print(x.shape)
print(x.dtype)
```

但 Tensor 相比 NumPy 最重要的两个能力是：

1. 可以运行在 GPU / accelerator 上
2. 可以通过 Autograd 自动计算梯度

因此在深度学习中，Tensor 不只是“存数据的数组”，还是 computation graph 中的节点。

---

## 2. Stable Softmax

对于 logits：

\[
z = [z_1,z_2,\dots,z_n]
\]

Softmax：

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

直接计算 `exp(z)` 可能产生数值溢出。

因此实际计算时先减去最大值：

\[
p_i =
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
\]

PyTorch 实现：

```python
def softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

减去最大值不会改变 Softmax 的最终结果，但能显著提高 numerical stability。

---

## 3. Cross Entropy

分类任务中，如果正确类别为 \(y\)，Cross Entropy 可以写成：

\[
L=-\log p_y
\]

PyTorch 中通常直接使用：

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

重要：

**不要先对 logits 手动做 Softmax。**

`CrossEntropyLoss` 接收的是 raw logits，内部等价于组合：

```text
LogSoftmax + NLLLoss
```

例如：

```python
logits = model(x)
loss = loss_fn(logits, labels)
```

而不是：

```python
probs = torch.softmax(logits, dim=-1)
loss = loss_fn(probs, labels)  # Wrong usage
```

---

# 4. Autograd

Autograd 是 PyTorch 自动求导系统。

例如：

```python
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

数学上：

\[
y=x^2
\]

所以：

\[
\frac{dy}{dx}=2x
\]

当 \(x=3\)：

\[
\frac{dy}{dx}=6
\]

因此：

```text
x.grad = 6
```

---

## 5. Computation Graph & Chain Rule

考虑：

\[
u=2x+1
\]

\[
y=u^2
\]

代码：

```python
x = torch.tensor(3.0, requires_grad=True)

u = 2 * x + 1
y = u ** 2

y.backward()

print(x.grad)
```

根据 chain rule：

\[
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
\]

其中：

\[
\frac{dy}{du}=2u
\]

\[
\frac{du}{dx}=2
\]

因此：

\[
\frac{dy}{dx}=4u
\]

当：

\[
x=3,\quad u=7
\]

得到：

\[
\frac{dy}{dx}=28
\]

PyTorch 的 `backward()` 本质上就是沿 computation graph 反向应用 chain rule。

---

# 6. Leaf Tensor vs Non-leaf Tensor

例如：

```python
x = torch.tensor(3.0, requires_grad=True)

u = 2 * x + 1
y = u ** 2
```

这里：

```text
x → leaf tensor
u → non-leaf tensor
y → non-leaf tensor
```

默认情况下，PyTorch 会保留 leaf tensor 的 `.grad`：

```python
y.backward()

print(x.grad)
```

但是：

```python
print(u.grad)
```

默认不会保存 intermediate tensor 的 gradient。

如果确实需要：

```python
u.retain_grad()
```

然后：

```python
y.backward()

print(u.grad)
```

注意：

> PyTorch 在 backward 时仍然会计算 intermediate gradient，只是默认不把它保存在 non-leaf tensor 的 `.grad` 中。

---

# 7. 从 Autograd 到 Machine Learning

一个简单模型：

\[
\hat y=wx+b
\]

Loss：

\[
L=(\hat y-y)^2
\]

例如：

```python
x = torch.tensor(2.0)

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = torch.tensor(10.0)

y_hat = w * x + b

loss = (y_hat - y) ** 2

loss.backward()
```

Forward：

\[
\hat y=3\times2+1=7
\]

\[
L=(7-10)^2=9
\]

Backward：

\[
\frac{\partial L}{\partial \hat y}
=
2(\hat y-y)
=
-6
\]

因为：

\[
\frac{\partial \hat y}{\partial w}=x=2
\]

所以：

\[
\frac{\partial L}{\partial w}
=
-6\times2
=
-12
\]

同时：

\[
\frac{\partial \hat y}{\partial b}=1
\]

所以：

\[
\frac{\partial L}{\partial b}
=
-6
\]

PyTorch 得到：

```text
w.grad = -12
b.grad = -6
```

这说明：

> Neural network training 中的 `loss.backward()` 和简单的 \(x^2\) 自动求导，本质上是同一件事。

---

# 8. Gradient Descent

Gradient descent：

\[
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
\]

其中：

- \(\theta\)：模型参数
- \(\eta\)：learning rate
- \(\nabla_\theta L\)：loss 对参数的 gradient

例如：

\[
w=3,\quad \frac{\partial L}{\partial w}=-12
\]

learning rate：

\[
\eta=0.1
\]

那么：

\[
w_{\text{new}}
=
3-0.1(-12)
=
4.2
\]

同理：

\[
b_{\text{new}}
=
1-0.1(-6)
=
1.6
\]

---

# 9. nn.Parameter

我们可以自己定义模型参数：

```python
from torch import nn

class MyLinearModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.tensor(3.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.w * x + self.b
```

`nn.Parameter` 是一种特殊 Tensor。

当它作为 `nn.Module` 的 attribute 时，会自动注册成模型参数。

因此：

```python
model = MyLinearModel()

print(list(model.parameters()))
```

可以找到：

```text
w
b
```

这非常重要，因为 optimizer 可以通过：

```python
model.parameters()
```

找到所有需要更新的参数。

---

# 10. nn.Module

`nn.Module` 是 PyTorch 神经网络模型的基础类。

通常结构：

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # define layers / parameters

    def forward(self, x):

        # define forward computation

        return output
```

使用模型：

```python
output = model(x)
```

而不是通常直接调用：

```python
model.forward(x)
```

`model(x)` 会通过 `nn.Module` 的调用机制执行 `forward()`，同时保留 PyTorch Module 的相关机制。

重要：

```python
model(x)
```

只表示 forward。

它**不会自动执行 backward**。

Backward 仍然需要：

```python
loss.backward()
```

---

# 11. nn.Linear

手动写：

```python
y_hat = w * x + b
```

可以使用 PyTorch 自带的：

```python
nn.Linear
```

例如：

```python
layer = nn.Linear(4, 8)
```

表示：

```text
in_features  = 4
out_features = 8
```

如果：

```python
x.shape == (5, 4)
```

那么：

```python
layer(x).shape == (5, 8)
```

可以理解为：

```text
5 samples
each sample has 4 features
        ↓
Linear(4, 8)
        ↓
5 samples
each sample has 8 features
```

`nn.Linear(4, 8)` 的 weight：

```python
layer.weight.shape
```

为：

```text
(8, 4)
```

对应：

\[
W\in\mathbb{R}^{8\times4}
\]

---

# 12. Optimizer

之前可以手动更新：

\[
w\leftarrow w-\eta\frac{\partial L}{\partial w}
\]

PyTorch 中可以使用 optimizer：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

然后：

```python
optimizer.step()
```

optimizer 会读取 parameter 的：

```text
parameter.grad
```

并更新 parameter。

因此可以理解为：

```text
loss.backward()
      ↓
parameter.grad
      ↓
optimizer.step()
      ↓
new parameters
```

---

# 13. Standard Training Loop

一个最基本的 PyTorch training loop：

```python
for epoch in range(num_epochs):

    model.train()

    optimizer.zero_grad()

    y_hat = model(x)

    loss = loss_fn(y_hat, y)

    loss.backward()

    optimizer.step()
```

整个过程：

```text
zero_grad
    ↓
forward
    ↓
loss
    ↓
backward
    ↓
optimizer.step
```

分别表示：

### `optimizer.zero_grad()`

清除上一轮累积的 gradient。

### `model(x)`

Forward pass，计算 prediction，同时建立 computation graph。

### `loss_fn(...)`

计算模型 prediction 与 target 之间的 loss。

### `loss.backward()`

沿 computation graph 使用 chain rule 计算：

\[
\nabla_\theta L
\]

并将结果保存到 parameter 的 `.grad`。

### `optimizer.step()`

利用 `.grad` 更新模型参数。

---

# 14. 为什么 Gradient 要清零？

PyTorch 默认会累积 gradient。

也就是说连续：

```python
loss.backward()
loss.backward()
```

gradient 会累加，而不是覆盖。

因此标准训练通常在每一轮使用：

```python
optimizer.zero_grad()
```

---

# 15. 一个样本为什么不能确定唯一的 Linear Model？

如果模型：

\[
\hat y=wx+b
\]

只有一个训练样本：

\[
x=2,\quad y=10
\]

那么只需要：

\[
2w+b=10
\]

例如：

\[
w=4,b=2
\]

可以。

但：

\[
w=3,b=4
\]

也可以。

甚至：

\[
w=5,b=0
\]

仍然可以。

所以一个 equation 无法唯一确定两个 unknown parameters。

这说明：

> Training loss 很低，不代表模型一定学到了真正的数据规律。

模型可能只是找到了一组能够解释当前 training data 的 parameters。

---

# 16. Learning a Linear Relationship

假设真实数据规律：

\[
y=2x+1
\]

训练数据：

```python
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])
```

模型：

```python
model = nn.Linear(1, 1)
```

训练：

```python
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    optimizer.zero_grad()

    y_hat = model(x)

    loss = loss_fn(y_hat, y)

    loss.backward()

    optimizer.step()
```

训练过程中应该看到：

\[
loss\rightarrow0
\]

同时：

\[
weight\rightarrow2
\]

\[
bias\rightarrow1
\]

optimizer 并不知道正确答案是：

```text
w = 2
b = 1
```

它只根据 gradient 不断寻找能够降低 loss 的方向。

---

# 17. train() vs eval() vs no_grad()

这三个概念不能混在一起。

## `model.train()`

将模型设置为 training mode。

会影响一些 layer，例如：

```text
Dropout
BatchNorm
```

例如 Dropout 在 training mode 下会随机丢弃 activation。

---

## `model.eval()`

将模型设置为 evaluation mode。

例如：

```python
model.eval()
```

会让 Dropout 停止随机 dropout，并改变 BatchNorm 的行为。

但是：

> `model.eval()` 不会关闭 Autograd。

---

## `torch.no_grad()`

关闭 gradient tracking：

```python
with torch.no_grad():
    y_hat = model(x)
```

常用于 inference / evaluation，可以避免构建不需要的 computation graph。

所以常见 evaluation：

```python
model.eval()

with torch.no_grad():
    y_hat = model(x)
```

可以记成：

```text
model.eval()
    ↓
改变模型行为

torch.no_grad()
    ↓
改变 Autograd 行为
```

---

# 18. Tiny MLP Classifier

一个简单的分类网络：

```python
class TinyClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits
```

网络结构：

```text
4 features
    ↓
Linear(4, 8)
    ↓
ReLU
    ↓
Linear(8, 3)
    ↓
3 logits
```

如果：

```python
x.shape == (5, 4)
```

表示：

```text
5 samples
4 features / sample
```

那么：

```python
logits = model(x)
```

得到：

```text
logits.shape == (5, 3)
```

表示：

```text
5 samples
3 class logits / sample
```

---

# 19. Classification Training

Labels：

```python
labels = torch.tensor([0, 2, 1, 0, 2])
```

Loss：

```python
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(logits, labels)
```

Optimizer：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

一次 training step：

```python
optimizer.zero_grad()

logits = model(x)

loss = loss_fn(logits, labels)

loss.backward()

optimizer.step()
```

对于：

```python
model.fc1 = nn.Linear(4, 8)
```

有：

```text
model.fc1.weight.shape
=
(8, 4)
```

因此：

```text
model.fc1.weight.grad.shape
=
(8, 4)
```

因为 gradient：

\[
\frac{\partial L}{\partial W}
\]

必须与参数：

\[
W
\]

具有相同 shape。

---

# 20. Mental Model

PyTorch 神经网络训练最核心的 mental model：

## Forward

```text
input
  ↓
model parameters
  ↓
computation graph
  ↓
prediction
  ↓
loss
```

数学上：

\[
\hat y=f_\theta(x)
\]

\[
L=L(\hat y,y)
\]

---

## Backward

```text
loss
  ↓
backpropagation
  ↓
chain rule
  ↓
parameter.grad
```

即计算：

\[
\nabla_\theta L
\]

---

## Update

```text
parameter.grad
      ↓
optimizer.step()
      ↓
new parameters
```

即：

\[
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
\]

---

# 21. The Big Picture

PyTorch training 可以压缩成：

\[
\boxed{
\text{Data}
\rightarrow
\text{Model}
\rightarrow
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
}
\]

代码上：

```python
optimizer.zero_grad()

prediction = model(x)

loss = loss_fn(prediction, target)

loss.backward()

optimizer.step()
```

其中：

```text
Tensor
    ↓
保存数据 / 参与 computation graph

nn.Parameter
    ↓
模型需要学习的 Tensor

nn.Module
    ↓
组织 parameters 和 layers

forward()
    ↓
定义模型如何计算

Autograd
    ↓
自动计算 gradients

Optimizer
    ↓
根据 gradients 更新 parameters
```

这条主线不会因为模型变成 CNN、RNN、Transformer 而改变。

模型会越来越复杂，但核心训练机制仍然是：

\[
\boxed{
Forward
\rightarrow
Loss
\rightarrow
Backward
\rightarrow
Update
}
\]

---

## Day 2 Takeaways

完成 Day 2 后，我应该能够解释：

- Tensor 与 NumPy ndarray 的核心区别
- `requires_grad=True` 的作用
- computation graph 是什么
- `loss.backward()` 做了什么
- leaf / non-leaf tensor 的区别
- `nn.Parameter` 为什么能成为模型参数
- `nn.Module` 的作用
- `model(x)` 与 `forward()` 的关系
- `nn.Linear(in_features, out_features)` 的 shape 变化
- `model.parameters()` 为什么能交给 optimizer
- `optimizer.zero_grad()` / `backward()` / `step()` 的关系
- `train()` / `eval()` / `no_grad()` 的区别
- 为什么 `CrossEntropyLoss` 前不需要手动 Softmax
- 一个标准 PyTorch training loop 的完整工作流程