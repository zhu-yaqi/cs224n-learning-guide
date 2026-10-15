#
>goal:Today I need to understand:
symbol -> context -> vector
count -> co-occurrence matrix -> SVD -> embedding
prediction -> Word2Vec -> embedding
global counts -> GloVe -> embedding

>embedding -> cosine similarity
embedding -> analogy
embedding -> failure cases
embedding -> bias
```
                Word Meaning
                     |
          -----------------------
          |                     |
      Count-based          Prediction-based
          |                     |
   co-occurrence              Word2Vec
          |                /            \
         SVD          Skip-gram         CBOW
          |
        GloVe  <--- uses global co-occurrence statistics

                     ↓
               dense embedding
                     ↓
      similarity / analogy / downstream tasks
                     ↓
      polysemy / bias / evaluation failures
```
## 1.lecture 1 notes
### (1) WordNet
  - 能表示词与词的包含关系
  - 能表示近义词
  - 太人工，不能表示词之间的细微差别，不能表示很多现代俚语
### (2) 词向量
  - 独热向量 one-hot vectors
  - 不能表示此之间的关联性，因为向量点积为0，相互正交
```
    motel=[0,0,0,0,0,1,0,0,0,0]
    hotel=[0,0,0,1,0,0,0,0,0,0]
```
  - 局部表示：每个词语在向量中只占一个位置
### (3) 词嵌入
```
每个词映射到高维空间，词向量的维数就是空间的维度，词语含义越相近，向量之间相似度越大，点积越大
```
  - 分布式表示：向量有一定相似度，每个词语在向量中占多个位置
## 2.Word2Vec
```
学习词向量的方法
```
### (1) Example windows and process for computing $P(w_{t+j} | w_t)$

> 窗口大小 window size = 2
> - center word（中心词，位置$t$）：**banking**
> - context words（上下文词，窗口内左右各2个）：`turning`, `into`, `crises`, `as`

句子片段：`... problems | turning | into | banking | crises | as ...`

目标：已知中心词 $w_t$，预测窗口内上下文词 $w_{t+j}$ 的条件概率 $P(w_{t+j} | w_t)$

窗口内4个预测任务：
- $P(w_{t-2} | w_t)$：$P(\text{turning} \mid \text{banking})$
- $P(w_{t-1} | w_t)$：$P(\text{into} \mid \text{banking})$
- $P(w_{t+1} | w_t)$：$P(\text{crises} \mid \text{banking})$
- $P(w_{t+2} | w_t)$：$P(\text{as} \mid \text{banking})$

> Skip-gram 核心思想：**给定中心词，预测窗口内周围上下文单词**
> window size=2：中心词左右各取2个词作为上下文。

### (2) Objective Function（Skip-gram）

任务：对文本中每个位置 $t=1,\dots,T$，给定中心词 $w_t$，预测固定窗口大小 $m$ 内的上下文单词。

#### <1>. 似然函数 Likelihood
$$
L(\theta)=\prod_{t=1}^T \prod_{\substack{-m\le j\le m\\j\neq0}} P(w_{t+j}\mid w_t;\theta)
$$
- $T$：整篇语料的单词总数
- $m$：上下文窗口大小（中心词左右各 $m$ 个词）
- $j\neq0$：跳过中心词本身，只取上下文词
- $\theta$：模型全部待学习参数集合（输入词矩阵 $W$ + 输出词矩阵 $W'$）

> 含义：联合概率乘积，代表**在当前参数 $\theta$ 下，观测到整段文本的概率**。

#### <2>. 目标函数：平均负对数似然 (average negative log likelihood)
$$
J(\theta)=-\frac1T\log L(\theta)
=-\frac1T\sum_{t=1}^T\sum_{\substack{-m\le j\le m\\j\neq0}} \log P(w_{t+j}\mid w_t;\theta)
$$

推导要点：
1. $\log$ 将乘积 $\prod$ 转为求和 $\sum$，简化求导，同时避免小数连乘造成数值下溢；
2. 负号：最大化似然 → 等价于**最小化损失**；
3. $\frac1T$：除以总词数 $T$，求**平均损失**，消除语料长度带来的数值影响，方便对比。

> 训练目标：**最小化目标函数 $J(\theta)$**，等价于最大化中心词预测上下文单词的预测准确率。

## 3. GloVe 词向量

### (1)核心定位

GloVe（Global Vectors）是**上下文无关的静态词向量**，结合了：

- LSA（矩阵分解）的**全局语料统计**

- Word2Vec（预测模型）的**语义拟合能力**

核心思想：**用词向量点积，拟合词的全局共现概率**。

### (2)完整训练流程（4步）

#### <1> 构建共现矩阵 X

- 遍历整个语料库，设置固定滑动窗口

- **Xij**：词 j 出现在词 i 上下文窗口的总次数（距离越远权重越低，权重为 1/d）

- 矩阵为稀疏矩阵，只记录共现过的词对

#### <2> 核心拟合公式

让词向量的点积 \+ 偏置，近似共现对数：
$$
w_i^\top \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}
$$
- $w_i$：中心词向量

- $\tilde{w}_j$：上下文词向量

- $b_i、\tilde{b}_j$：偏置项

#### <3> 加权损失函数（核心）

采用加权最小二乘损失，解决高低频词不均衡问题：
$$
J=\sum_{i,j} f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j -\log X_{ij}\right)^2
$$

权重函数 **f\(x\)** 作用：


- 低频词：降低权重，减少噪声影响

- 高频词：权重封顶，避免主导训练

#### <4> 输出最终词向量

训练完成后，合并两套向量提升效果：

**最终词向量 = 中心词向量 \+ 上下文词向量**

### 三、核心特性

1. **静态向量**：一词一向量，上下文无关，无法解决一词多义

2. **全局信息**：基于全量语料共现统计，优于Word2Vec局部窗口统计

3. **语义推理强**：擅长类比任务（king \- man \+ woman ≈ queen）


## 五、优缺点总结

### 优点

- 融合全局统计信息，语义表达更稳定

- 训练高效，仅计算非零共现词对

- 词类比、相似度任务效果优异

### 缺点

- 静态词向量，**不支持一词多义**

- 无法根据上下文动态调整词向量（区别于BERT等动态模型）

