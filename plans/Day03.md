# CS224N Day 3 学习文档：Word Vectors + Assignment 1

> **对应 24 天计划：Day 3**\
> **建议总时长：5.5--6.5 小时**\
> **主线：Word Vectors → Co-occurrence → SVD → Word2Vec → GloVe → Cosine
> Similarity → Evaluation → Polysemy / Failure Cases / Bias → Assignment
> 1**
>
> 今天的目标不是"把视频看完"，而是建立第一套完整的 NLP 表示学习 mental
> model，并完成/推进 2026 Assignment 1。\
> **每遇到 NumPy 矩阵操作，都写清楚输入 shape、操作 axis、输出 shape。**

------------------------------------------------------------------------

## 0. 今天结束时必须达到什么水平

Day 3 结束时，你应该能够不看资料解释下面这些问题：

1.  为什么 one-hot vector 不能自然表达词义相似度？
2.  什么是 distributional semantics（分布式语义）？
3.  "You shall know a word by the company it keeps" 对 word vectors
    意味着什么？
4.  word vector / word embedding / distributed representation
    分别是什么意思？
5.  context window 是什么？窗口大小改变会影响什么？
6.  co-occurrence matrix 是怎么构造的？
7.  为什么原始 co-occurrence vector 维度太高、太稀疏？
8.  SVD 为什么可以把 `[V,V]` 的共现矩阵压到 `[V,k]`？
9.  truncated SVD 保留的是什么？为什么可以视为低维语义表示？
10. Word2Vec 到底在"预测"什么？
11. Skip-gram 与 CBOW 的方向有什么区别？
12. 为什么 Word2Vec 中一个词通常有 center vector 和 outside/context
    vector 两套表示？
13. `u_o^T v_c` 表示什么？shape 是什么？
14. softmax 如何把全词表的 scores 变成 `P(o|c)`？
15. Word2Vec 的 negative log-likelihood objective 在优化什么？
16. 为什么 naïve softmax 在大词表上很贵？
17. negative sampling 的核心思想是什么？
18. GloVe 与 Word2Vec / count-based 方法的关系是什么？
19. cosine similarity 为什么比单纯 Euclidean distance 更适合比较
    embedding 方向？
20. intrinsic evaluation 与 extrinsic evaluation 有什么区别？
21. word analogy 为什么可以写成向量算术？
22. 为什么 antonym 有时反而比 synonym 更"近"？
23. 为什么一个静态 embedding 很难同时表示一个多义词的多个词义？
24. embedding bias 从哪里来？为什么它不是一个纯技术小问题？
25. 为什么低维 2D 图只能帮助观察，而不能证明高维 embedding 的全部性质？

如果其中任何一条只能"感觉懂"，但不能自己讲清楚，今天还没有真正结束。

------------------------------------------------------------------------

# 1. 今天只使用这些资料

## A. 2026 CS224N 官方课程页 ------ 版本基准

**用途：确认当前课程结构、2026 slides / notes / Assignment 1。**

https://web.stanford.edu/class/cs224n/

进入：

-   Week 1 → **History of NLP**
-   Week 1 → **Word Vectors**
    -   slides
    -   notes 1
    -   notes 2
-   Assignment 1 → **code**

> 重要：作业以 **Winter 2026 当前官网的 Assignment 1 code**
> 为准。不要因为网上容易搜到 2024/2025 preview，就改做旧版作业。

------------------------------------------------------------------------

## B. 2024 Stanford 官方公开视频 Lecture 1

**Stanford CS224N Spring 2024 --- Lecture 1: Intro and Word Vectors**

https://www.youtube.com/watch?v=DzpHeXVSC5I

今天重点看：

-   Human language and word meaning
-   Word2Vec introduction
-   Word2Vec objective
-   Word2Vec gradients：今天先理解梯度在"推动什么"，详细矩阵求导留到 Day
    4
-   Optimization basics
-   Looking at word vectors

建议速度：

-   概念：`1.25×`
-   已经非常熟悉的优化部分：`1.5×`
-   数学公式第一次出现时暂停，不要倍速硬冲

------------------------------------------------------------------------

## C. 2024 Stanford 官方公开视频 Lecture 2

**Stanford CS224N Spring 2024 --- Lecture 2: Word Vectors and Language
Models**

https://www.youtube.com/watch?v=nBor4jfWetQ

今天重点看：

-   Review of Word2Vec
-   More on Word2Vec
-   Count-based representations
-   Evaluating word vectors
-   Word senses

后面的 classification / neural networks 当作 **Day 4
预览**，今天听懂"为什么 embedding
可以作为神经网络输入"即可，不要求推导。

------------------------------------------------------------------------

## D. 2026 Word Vectors slides

课程页 Week 1 → Word Vectors → slides。

2026 slides 共 44 页左右，今天建议：

-   **重点精读：slides 4--43（PDF 页码可能与幻灯片显示页码差 1）**
-   Course organization 可快速略过
-   不要略过最后的 evaluation 部分

核心区间：

  Slides 内容                            重点
  -------------------------------------- ----------------------------------------------
  Meaning / WordNet / discrete symbols   为什么旧表示不够
  Distributional semantics               整个 embedding 思想的起点
  Word2Vec overview                      center / outside / window
  Objective + softmax                    Word2Vec 数学核心
  Gradient descent / SGD                 如何训练
  Skip-gram / CBOW                       两种方向
  Negative sampling                      为什么需要近似
  Co-occurrence matrix                   count-based 表示
  SVD                                    A1 直接使用
  GloVe                                  global count + vector geometry
  Evaluation                             intrinsic / extrinsic / analogy / similarity

------------------------------------------------------------------------

## E. 2026 Notes

课程页 Week 1 → Word Vectors：

-   **notes 1：Introduction / Word2Vec**
-   **notes 2：GloVe / Evaluation / Training**

今天不要逐字精读。视频和 slides 建立框架后，用 notes 查公式和补遗漏。

------------------------------------------------------------------------

## F. 2026 Assignment 1

课程页：

https://web.stanford.edu/class/cs224n/

进入 Week 1 → Assignment 1 → `code`。

今天的原则：

> **先读题 → 自己写 → 跑 sanity check → 再 debug。不要搜索公开
> solution。**

A1 的核心训练价值不是"拿到一个漂亮 embedding 图"，而是把 Day 1 的 NumPy
真正用于 NLP：

-   vocabulary / index mapping
-   co-occurrence
-   matrix construction
-   SVD / truncated SVD
-   row normalization
-   broadcasting
-   cosine similarity
-   GloVe exploration
-   analogy
-   failure analysis
-   bias analysis

------------------------------------------------------------------------

# 2. Day 3 总时间表

  Block      内容                                                                时间
  ---------- ---------------------------------------------------- -------------------
  0          快速预习 + 建立问题地图                                           15 min
  1          词义表示与 distributional semantics                               35 min
  2          Word2Vec / Skip-gram / objective / softmax                        55 min
  3          Negative sampling + CBOW + optimization                           30 min
  4          Co-occurrence + SVD + GloVe                                       45 min
  5          Cosine / evaluation / polysemy / bias                             35 min
  6          Assignment 1：Count-based 部分                                70--85 min
  7          Assignment 1：GloVe / similarity / failures / bias            70--85 min
  8          闭卷复盘 + GitHub notes                                           30 min
  **合计**                                                          **约 5.8--6.5 h**

如果 A1 下载模型很慢，不要坐着等：下载期间直接做 Block 8 的笔记。

------------------------------------------------------------------------

# 3. Block 0 --- 先建立今天的问题地图

**耗时：15 min**

先打开：

1.  2026 Word Vectors slides
2.  2026 A1 notebook / code
3.  `notes/02_word_vectors.md`

暂时不要写答案，只在 notes 顶部写：

``` text
Today I need to understand:

symbol -> context -> vector
count -> co-occurrence matrix -> SVD -> embedding
prediction -> Word2Vec -> embedding
global counts -> GloVe -> embedding

embedding -> cosine similarity
embedding -> analogy
embedding -> failure cases
embedding -> bias
```

然后画今天最重要的一张总图：

``` text
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

### Block 0 验收

你必须知道今天不是在学"几个 embedding API"，而是在回答：

> **机器怎样把离散的词变成可以计算语义关系的连续向量？**

------------------------------------------------------------------------

# 4. Block 1 --- 从离散符号到 Distributional Semantics

**耗时：35 min**

## 资料

先看 2024 Lecture 1 的：

-   Human language and word meaning
-   Word2Vec introduction 前半

同时对照 2026 Word Vectors slides：

-   meaning
-   WordNet
-   discrete symbols / one-hot
-   representing words by context
-   word vectors

------------------------------------------------------------------------

## 4.1 Denotational semantics

传统思路：

``` text
word / signifier  <->  concept / signified
```

例如：

``` text
"tree" -> 某个现实/概念中的“树”
```

问题是计算机不能直接操作"概念本身"，必须把它编码成某种数学表示。

------------------------------------------------------------------------

## 4.2 WordNet 思路及局限

理解 WordNet 这种人工词汇资源：

-   synonym
-   hypernym
-   lexical relations

必须知道局限：

-   人工维护成本高
-   新词 / 新义更新困难
-   同义词并不意味着所有语境完全可替换
-   很难直接得到连续、细粒度 similarity
-   词的社会含义、语用、connotation 很难由简单词典关系表达

今天不需要学习 WordNet API。

------------------------------------------------------------------------

## 4.3 One-hot representation

若词表大小：

``` text
|V| = 50,000
```

一个词：

``` text
x ∈ R^50000
```

只有一个位置为 1。

### 必须回答

假设：

``` text
hotel = [0,0,1,0,...]
motel = [0,0,0,1,...]
```

为什么它们在语义上很接近，但：

``` text
hotel · motel = 0
```

？

答案的关键不是公式，而是：

> one-hot 的坐标只表示"身份"，没有编码词之间的语义关系。

------------------------------------------------------------------------

## 4.4 Distributional semantics

核心思想：

> 一个词的意义，可以从它经常出现的上下文中学习。

设语料：

``` text
... government debt problems turning into banking crises ...
... unified banking regulation ...
... banking system ...
```

那么 `banking` 周围经常出现的词会逐渐定义它的表示。

### 必须掌握

-   target / center word
-   context word / outside word
-   context window
-   corpus
-   vocabulary
-   word type vs word token

### 自测

解释：

``` text
window_size = 2

the quick brown fox jumps
          ↑
        center
```

`brown` 的 context 是哪些词？

------------------------------------------------------------------------

# 5. Block 2 --- Word2Vec：从上下文预测学习词向量

**耗时：55 min**

## 资料

2024 Lecture 1：

-   Word2Vec introduction
-   objective function
-   objective gradients
-   looking at word vectors

2026 slides：

-   Word2Vec overview
-   objective
-   prediction function
-   softmax
-   parameters

------------------------------------------------------------------------

## 5.1 Word2Vec 不是一个"固定向量表"

正确 mental model：

``` text
初始化随机 vectors
        ↓
从 corpus 产生 center-context training pairs
        ↓
根据 vectors 计算预测概率
        ↓
计算 loss
        ↓
更新 vectors
        ↓
大量训练后
        ↓
vectors 本身变成有语义结构的表示
```

最终我们要的不是预测器本身，而是训练过程中学到的 embedding。

------------------------------------------------------------------------

## 5.2 Skip-gram

给 center word：

``` text
c
```

预测窗口中的 outside word：

``` text
o
```

即：

\[ P(o\|c) \]

例如：

``` text
... problems turning into banking crises as ...
                         ↑ center = banking
```

如果窗口覆盖附近词，就产生多个训练 pair。

### 必须会区分

**Skip-gram**

``` text
center -> context
```

**CBOW**

``` text
context bag -> center
```

今天不需要实现完整 Word2Vec，但必须能说出方向。

------------------------------------------------------------------------

## 5.3 为什么一个词有两个向量

每个词 `w`：

``` text
v_w = center vector
u_w = outside/context vector
```

设 embedding dimension：

``` text
D = 300
```

则：

``` text
v_c.shape = (D,)
u_o.shape = (D,)
```

全体 outside vectors：

``` text
U.shape = (|V|, D)
```

注意：深度学习库通常把 embedding 作为"每行一个词"。

------------------------------------------------------------------------

## 5.4 Score

center 与 outside 的匹配分数：

\[ u_o\^T v_c \]

shape：

``` text
(D,) · (D,) -> scalar
```

对整个词表：

``` text
U          : [V, D]
v_c        : [D]
U @ v_c    : [V]
```

这就是你 Day 1 学的 matrix multiplication 在 NLP 中第一次真正出现。

------------------------------------------------------------------------

## 5.5 Softmax probability

\[ P(o\|c) = `\frac{\exp(u_o^T v_c)}`{=tex}
{`\sum`{=tex}\_{w`\in `{=tex}V}`\exp`{=tex}(u_w\^T v_c)} \]

逐步写 shape：

``` text
U              [V,D]
v_c            [D]
scores=U@v_c   [V]
softmax        [V]
P(o|c)         scalar
```

### 必须理解

dot product 越大：

``` text
u_o^T v_c ↑
    ↓
score ↑
    ↓
softmax probability ↑
```

训练会推动真实 context 的向量与 center 向量更匹配。

------------------------------------------------------------------------

## 5.6 Objective

对 corpus 中每个 center word，预测窗口中的 context words。

negative log likelihood：

\[ J(`\theta`{=tex}) = -`\frac{1}{T}`{=tex} `\sum`{=tex}*{t=1}\^{T}
`\sum`{=tex}*{`\substack{-m\le j\le m\\j\ne0}`{=tex}}
`\log `{=tex}P(w\_{t+j}\|w_t) \]

今天必须理解每个符号：

-   `T`：corpus 中位置数
-   `t`：当前 center 的位置
-   `m`：window radius
-   `j`：相对 center 的偏移
-   `w_t`：center
-   `w_{t+j}`：outside word
-   `θ`：所有 word vectors

不要死背整条式子。你要能把它翻译成中文：

> 对语料中的每个中心词，让它给真实邻居更高概率。

------------------------------------------------------------------------

## 5.7 梯度今天学到什么程度

Day 4 才正式做矩阵求导，所以今天只要求理解：

如果真实 outside word 是 `o`：

-   模型给 `o` 概率太低 → loss 大
-   gradient 会调整 vectors
-   让真实 center-context pair 的 score 变大
-   同时让错误词的相对概率下降

今天不要花 40 分钟死磕完整 derivative。

但要记住：

``` text
forward:
vectors -> scores -> softmax -> loss

backward:
loss -> gradients of word vectors

update:
vectors move in embedding space
```

这与你 Day 2 的 PyTorch mental model完全一致。

------------------------------------------------------------------------

# 6. Block 3 --- Naïve Softmax、Negative Sampling、CBOW、SGD

**耗时：30 min**

## 资料

2026 Word Vectors slides：

-   Word2Vec algorithm family
-   Skip-gram vs CBOW
-   negative sampling
-   stochastic gradients

------------------------------------------------------------------------

## 6.1 Naïve softmax 的成本

计算：

``` text
U @ v_c
```

需要与整个 vocabulary 的所有 outside vectors 做比较。

如果：

``` text
|V| = 500,000
```

每个 training pair 都对 50 万词归一化，成本很高。

------------------------------------------------------------------------

## 6.2 Negative Sampling

核心思想：

不要每次都与整个词表比较。

把任务变成：

``` text
真实 pair:
(center, true_context) -> 1

随机 noise pairs:
(center, random_word_1) -> 0
(center, random_word_2) -> 0
...
```

目标：

\[ -`\log `{=tex}`\sigma`{=tex}(u_o\^T v_c)
-`\sum`{=tex}\_{k}`\log`{=tex}`\sigma`{=tex}(-u_k\^T v_c) \]

今天必须理解：

``` text
positive pair -> dot product should increase
negative pair -> dot product should decrease
```

以及：

``` text
softmax:
compare against entire vocabulary

negative sampling:
compare against true word + small number of sampled noise words
```

------------------------------------------------------------------------

## 6.3 Sigmoid

\[ `\sigma`{=tex}(x)=`\frac{1}{1+e^{-x}}`{=tex} \]

这里只需理解：

``` text
large positive x -> σ(x) ≈ 1
large negative x -> σ(x) ≈ 0
```

------------------------------------------------------------------------

## 6.4 Sampling distribution

课程 slides 提到经典 Word2Vec negative sampling 常使用 unigram frequency
的 `3/4` 次幂调整采样。

今天记住直觉即可：

> 不直接按原始词频采样；通过调整分布，避免极高频词过度主导，同时让较低频词有更多机会成为
> negative samples。

------------------------------------------------------------------------

## 6.5 SGD / sparse updates

Word2Vec 的一个重要计算特点：

一次 window 只涉及少量词。

negative sampling 后，只需更新：

-   center vector
-   true outside vector
-   sampled negative vectors

所以梯度更新可以很 sparse。

这会在以后学习 embedding lookup、稀疏参数更新时再次出现。

------------------------------------------------------------------------

# 7. Block 4 --- Co-occurrence Matrix、SVD 与 GloVe

**耗时：45 min**

## 资料

2026 slides：

-   Why not capture co-occurrence counts directly?
-   window-based co-occurrence matrix
-   co-occurrence vectors
-   SVD
-   hacks to X
-   GloVe

同时开始对照 A1 Part 1。

------------------------------------------------------------------------

## 7.1 Co-occurrence matrix

词表大小：

``` text
V = number of unique words
```

构造：

``` text
X.shape = [V,V]
```

其中：

``` text
X[i,j]
```

表示词 `i` 与词 `j` 在指定 context window 内共同出现的次数。

------------------------------------------------------------------------

## 7.2 Window

例如：

``` text
I like deep learning
```

window size = 1。

对 `like`：

``` text
left  = I
right = deep
```

因此：

``` text
count(like, I) += 1
count(like, deep) += 1
```

### 边界必须处理

句首 / 句尾的 window 会自然变小。

A1 中如果使用 `<START>` / `<END>`，它们也是 vocabulary 中的
token，并参与 count。

------------------------------------------------------------------------

## 7.3 Shape mental model

``` text
V = len(corpus_words)

M.shape = (V,V)

word2ind:
word -> row/column index
```

例如：

``` python
i = word2ind[center]
j = word2ind[context]
M[i, j] += 1
```

如果定义的是对称 window，通常对应的统计最终会体现双向共现；实现时严格按
A1 题意，不要凭印象额外重复计数。

------------------------------------------------------------------------

## 7.4 为什么 raw co-occurrence 不够

问题：

-   `[V,V]` 维度巨大
-   sparse
-   storage / downstream model 不方便
-   高频 function words 容易主导 count
-   raw frequency scale 不一定等价于语义重要性

------------------------------------------------------------------------

## 7.5 SVD

\[ X = U`\Sigma `{=tex}V\^T \]

如果：

``` text
X: [V,V]
```

完整 SVD：

``` text
U       [V,V]
Sigma   [V,V]
V^T     [V,V]
```

保留 top-k components：

``` text
embedding: [V,k]
```

A1 使用 truncated SVD，目标不是让你手写 SVD 算法，而是理解：

> 用低秩结构保留共现矩阵中最重要的变化方向，把每个词压成 k 维 dense
> vector。

------------------------------------------------------------------------

## 7.6 Rank-k approximation

你应该能解释：

``` text
[V,V] huge sparse representation
            ↓
       top-k structure
            ↓
[V,k] compact dense representation
```

`k` 越小：

-   压缩更强
-   信息损失更多
-   更容易可视化

`k=2`：

-   适合画图
-   **不代表实际 embedding 最佳维度就是 2**

------------------------------------------------------------------------

## 7.7 对 row 做 normalization

若：

``` text
M_reduced.shape = [V,k]
```

row norm：

``` python
lengths = np.linalg.norm(M_reduced, axis=1)
```

shape：

``` text
[V]
```

为了除回每一行：

``` python
M_normalized = M_reduced / lengths[:, np.newaxis]
```

shape：

``` text
M_reduced             [V,k]
lengths[:,None]       [V,1]
broadcast             [V,k]
result                [V,k]
```

这是 Day 1 broadcasting 的直接复习。

------------------------------------------------------------------------

## 7.8 Count-based 的常见改进直觉

2026 slides 还提到 raw counts 的一些问题与处理思路：

-   log frequency
-   clipping / threshold
-   ignore function words
-   distance-weighted / ramped windows
-   correlation-based statistics
-   negative values clipping 等

今天不要求实现，但不能不知道：

> "co-occurrence + SVD" 并不是只要把原始计数扔进 SVD 就自动完美。

------------------------------------------------------------------------

## 7.9 GloVe

GloVe = **Global Vectors**。

关键直觉：

> Word2Vec 从局部预测任务学习表示；GloVe
> 更直接利用全局词-词共现统计，并让词向量几何结构编码共现概率关系。

你今天不需要从头推 GloVe loss 的所有细节，但必须知道：

-   使用 global co-occurrence statistics
-   学 dense vectors
-   词义关系可以体现在 vector differences 中
-   它不是"纯 SVD"，也不是"完全不看 counts 的预测模型"

------------------------------------------------------------------------

# 8. Block 5 --- Similarity、Evaluation、Polysemy、Failure Cases、Bias

**耗时：35 min**

## 资料

2024 Lecture 2：

-   evaluating word vectors
-   word senses

2026 slides：

-   intrinsic / extrinsic
-   analogies
-   similarity
-   GloVe visualization

A1 Part 2 与这部分高度重合。

------------------------------------------------------------------------

## 8.1 Cosine similarity

\[ `\cos`{=tex}(p,q) = `\frac{p\cdot q}{\|p\|\|q\|}`{=tex} \]

范围：

``` text
[-1,1]
```

重点是方向而非绝对长度。

### Shape

``` text
p [D]
q [D]

p*q          [D]
sum          scalar
norm(p)      scalar
norm(q)      scalar
cosine       scalar
```

------------------------------------------------------------------------

## 8.2 Cosine distance

常见定义：

\[ d\_{`\cos`{=tex}}=1-`\cos`{=tex}(p,q) \]

因此：

``` text
similarity high -> distance low
```

A1 里不要把 similarity 与 distance 方向写反。

------------------------------------------------------------------------

## 8.3 Analogy

经典结构：

``` text
man : grandfather :: woman : ?
```

向量关系近似：

\[ x `\approx `{=tex}w + g - m \]

一般形式：

``` text
a : b :: c : d
```

可寻找：

\[ d `\approx `{=tex}b-a+c \]

注意：

> analogy 能工作并不意味着 embedding
> 真的拥有完美的"逻辑推理"。它反映的是某些语义/句法关系在空间中近似线性。

------------------------------------------------------------------------

## 8.4 Intrinsic evaluation

直接评 embedding：

-   word similarity
-   analogy
-   与 human judgments 的 correlation

优点：

-   快
-   便宜
-   容易诊断表示本身

缺点：

-   intrinsic 分数高不保证真实 downstream task 一定更好

------------------------------------------------------------------------

## 8.5 Extrinsic evaluation

把 embedding 放进真实任务：

-   NER
-   sentiment
-   QA
-   classification
-   其他 downstream task

然后看最终任务指标是否提高。

问题：

-   慢
-   expensive
-   很难区分到底是 embedding 还是系统其他部分造成变化

------------------------------------------------------------------------

## 8.6 Polysemy / homonymy

静态 word embedding 的典型限制：

``` text
bank = river bank
bank = financial bank
```

传统 static embedding 往往：

``` text
one word type -> one vector
```

所以多个 sense 被压进同一个表示。

这会产生：

-   dominant sense 覆盖 rare sense
-   nearest neighbors 只表现一个常见义项
-   多个义项被混合

这正是后来 contextual embeddings 的重要动机之一。

------------------------------------------------------------------------

## 8.7 Synonym / antonym failure

为什么：

``` text
happy
sad
```

可能比：

``` text
happy
cheerful
```

更接近？

因为 distributional semantics 学的是：

``` text
similar contexts
```

而 antonyms 经常出现在非常相似的句法和语义环境：

``` text
I feel happy.
I feel sad.
```

因此：

> distributional similarity ≠ synonymy。

这是今天非常重要的结论。

------------------------------------------------------------------------

## 8.8 Bias

Embedding 从 corpus 学统计规律，因此 corpus 中存在的：

-   stereotype
-   historical inequality
-   representation imbalance
-   offensive association

都可能进入 vector geometry。

必须理解：

``` text
data distribution
      ↓
co-occurrence / training signal
      ↓
embedding geometry
      ↓
similarity / ranking / downstream decisions
```

不要把 bias 简化成"模型故意歧视"。

A1 要求你观察、解释并思考 mitigation。

------------------------------------------------------------------------

# 9. Block 6 --- Assignment 1：Count-Based Word Vectors

**耗时：70--85 min**

> 下面按 A1 的知识结构推进。实际题号以你下载到的 **2026 A1** 为准；如果
> 2026 notebook 的题号与旧 preview 不同，不要强行按旧题号操作。

------------------------------------------------------------------------

## Step 6.1 --- 先完整读 README / notebook 开头

**5--10 min**

确认：

-   dependencies
-   dataset
-   starter functions
-   tests
-   output requirements
-   是否有模型下载
-   当前 Python 环境

不要一打开 notebook 就从第一个 TODO 盲写。

------------------------------------------------------------------------

## Step 6.2 --- Distinct vocabulary

**10 min**

目标：

``` text
corpus: List[List[str]]
        ↓
unique words
        ↓
sorted vocabulary
```

检查：

``` text
len(corpus_words) == num_corpus_words
```

理解：

``` text
token = 某次出现
type  = 不重复词项
```

------------------------------------------------------------------------

## Step 6.3 --- Co-occurrence matrix

**20--25 min**

你写代码前先写 shape：

``` text
V = number of distinct words

M.shape = [V,V]
```

逻辑：

``` text
for each document
    for each center position
        determine valid left/right boundary
        for each context position in window
            increment count
```

### 必查 bug

-   是否把 center 自己算进去？
-   句子边界是否越界？
-   window 是 radius 还是总长度？
-   `<START>` / `<END>` 是否按题意处理？
-   word → index 是否一致？
-   count dtype 是否合理？

------------------------------------------------------------------------

## Step 6.4 --- Reduce to k dimensions

**10 min**

输入：

``` text
M [V,V]
```

输出：

``` text
M_reduced [V,k]
```

使用题目指定的 `TruncatedSVD` / starter API。

不要手写 SVD。

你必须在旁边写：

``` text
[V,V] -> [V,k]
```

------------------------------------------------------------------------

## Step 6.5 --- Plot embeddings

**10 min**

如果：

``` text
M_reduced.shape = [V,2]
```

取指定 words 的 row：

``` text
word -> word2ind[word] -> M_reduced[index]
```

得到：

``` text
[x,y]
```

画 scatter + labels。

### 不要过度解读

2D 是强压缩。

"两个词在图上没有靠得很近"不代表高维 embedding 中一定不相似。

------------------------------------------------------------------------

## Step 6.6 --- Co-occurrence plot analysis

**15--20 min**

至少观察：

-   哪些词 cluster
-   哪些你以为会 cluster，但没有
-   为什么可能出现这种情况

可能原因：

-   corpus size
-   window definition
-   frequency
-   dimensionality reduction
-   2D projection information loss
-   polysemy
-   domain-specific usage

不要只写：

``` text
because the model is imperfect
```

要指出可能的统计机制。

------------------------------------------------------------------------

# 10. Block 7 --- Assignment 1：GloVe、Cosine、Analogy、Failure、Bias

**耗时：70--85 min**

------------------------------------------------------------------------

## Step 7.1 --- 加载 pretrained GloVe

**5--15 min（下载时间另算）**

下载期间继续写 notes。

理解：

``` text
pretrained embedding matrix
        ↓
word -> vector
        ↓
similarity / analogy analysis
```

你今天不是训练 GloVe，而是在分析它。

------------------------------------------------------------------------

## Step 7.2 --- GloVe visualization

**10 min**

与自己的 co-occurrence + SVD 图比较：

观察：

-   similarity
-   difference
-   clusters
-   geometry

解释差异时考虑：

-   training corpus 更大
-   embedding dimension 更高
-   GloVe objective 不同
-   训练统计更丰富
-   2D projection 本身有损

------------------------------------------------------------------------

## Step 7.3 --- Cosine similarity

**10 min**

在 notebook 旁边写公式：

\[ `\cos`{=tex}(p,q)=`\frac{p^Tq}{\|p\|\|q\|}`{=tex} \]

确保你知道 library 的：

``` text
most_similar
distance
```

返回的是 similarity 还是 distance。

------------------------------------------------------------------------

## Step 7.4 --- Multiple meanings

**10 min**

尝试 polysemous / homonymous words。

你的分析必须回答：

1.  哪个词有多个 meaning？
2.  top neighbors 是否同时出现多个 sense？
3.  如果只出现一个 sense，为什么？

重点结论：

``` text
static word embedding
one word type -> one vector
```

------------------------------------------------------------------------

## Step 7.5 --- Synonym vs antonym failure

**10 min**

找：

``` text
w1, w2 = synonyms
w1, w3 = antonyms
```

但：

``` text
distance(w1,w3) < distance(w1,w2)
```

解释不能停在"embedding 不准"。

要写：

> antonyms 可能共享大量 context，因此 distributional similarity 会很高。

------------------------------------------------------------------------

## Step 7.6 --- Analogies

**10--15 min**

理解：

``` text
positive=[a, y]
negative=[x]
```

对应向量：

``` text
a + y - x
```

然后找：

-   一个成功 analogy
-   一个失败 analogy

失败分析考虑：

-   relation 不线性
-   corpus bias
-   word frequency
-   polysemy
-   morphology
-   named entities
-   rare words
-   nearest-neighbor geometry

------------------------------------------------------------------------

## Step 7.7 --- Bias analysis

**15 min**

完成 notebook 指定 bias experiments。

你的 reflection 至少写：

### Bias source

``` text
training corpus reflects real-world text
        ↓
social/historical associations appear in co-occurrence
        ↓
embedding learns these associations
```

### Risk

embedding 可能被 downstream system 使用：

``` text
ranking
classification
retrieval
recommendation
matching
```

从而传播不希望的 association。

### Mitigation 思路

今天只需要知道存在多种层次：

-   data curation / balancing
-   debiasing representations
-   task-specific constraints
-   evaluation / auditing
-   downstream mitigation

不要认为"把某一个 bias direction 删除"就自动解决所有偏见。

------------------------------------------------------------------------

# 11. 每一个 NumPy 操作都做 Shape Audit

这是今天必须执行的规则。

在 A1 每个核心函数旁边写类似：

``` python
# corpus_words: [V]
# M: [V, V]
# M_reduced: [V, k]
# row_norms: [V]
# row_norms[:, None]: [V, 1]
# M_normalized: [V, k]
```

遇到：

``` python
M / lengths[:, np.newaxis]
```

必须先回答：

``` text
[V,k] / [V,1]
        ↓ broadcasting
[V,k]
```

遇到 word vector：

``` text
embedding_matrix [V,D]
index            scalar
embedding[index] [D]
```

这件事比你今天记住所有论文年份都重要。

------------------------------------------------------------------------

# 12. 今天容易混淆的 12 个点

## 1. Word2Vec ≠ GloVe

都是 static word embeddings，但训练思想不同。

------------------------------------------------------------------------

## 2. Word2Vec ≠ "直接数 co-occurrence 然后 SVD"

Word2Vec 是 prediction-based objective。

------------------------------------------------------------------------

## 3. GloVe 也不是简单 raw-count SVD

它通过特定 objective 利用 global co-occurrence statistics 学 vectors。

------------------------------------------------------------------------

## 4. Similar context ≠ synonym

antonyms 也可能 context 很像。

------------------------------------------------------------------------

## 5. Dot product ≠ cosine similarity

\[ p\^Tq \]

会受到 vector magnitude 影响。

cosine 会做 norm normalization。

------------------------------------------------------------------------

## 6. SVD 的 `k=2` 只是为了可视化

实际 embedding 通常远高于 2 维。

------------------------------------------------------------------------

## 7. 2D plot 不是 ground truth

降维会损失信息。

------------------------------------------------------------------------

## 8. Window size 是超参数

较小 window 往往更强调局部/句法关系；较大 window
更容易捕获更宽的语义/主题信息。不要把这个当绝对定律，但要理解
trade-off。

------------------------------------------------------------------------

## 9. Static embedding 对多义词有天然限制

一个 token type 通常对应一个固定 vector。

------------------------------------------------------------------------

## 10. Word analogy 不是严格代数规律

只是 embedding geometry 中经常出现的近似线性结构。

------------------------------------------------------------------------

## 11. Bias 不是一个"单独坏词"的问题

它可能是整个 vector space 的统计结构。

------------------------------------------------------------------------

## 12. Day 3 不需要完整推导 Word2Vec gradient

今天理解 objective 和梯度方向；Day 4 再系统做 derivative / matrix
calculus。

------------------------------------------------------------------------

# 13. Block 8 --- 30 分钟闭卷复盘

**耗时：30 min**

关闭视频、slides、notebook。

在 `notes/02_word_vectors.md` 中闭卷写下面内容。

------------------------------------------------------------------------

## 13.1 一页 Mental Model

``` text
Discrete word
    ↓
contexts in corpus
    ↓
statistical learning signal
    ↓
dense vector
    ↓
geometry encodes some linguistic regularities
    ↓
similarity / analogy / downstream features
```

------------------------------------------------------------------------

## 13.2 Count-based pipeline

``` text
corpus
  ↓
vocabulary
  ↓
co-occurrence matrix [V,V]
  ↓
Truncated SVD
  ↓
embedding [V,k]
  ↓
normalize
  ↓
cosine / visualization
```

------------------------------------------------------------------------

## 13.3 Word2Vec pipeline

``` text
corpus
  ↓
center-context pairs
  ↓
embedding lookup
  ↓
dot products
  ↓
softmax / negative sampling
  ↓
loss
  ↓
gradient update
  ↓
learned embeddings
```

------------------------------------------------------------------------

## 13.4 必须闭卷写出的公式

### Cosine similarity

\[ `\cos`{=tex}(p,q)=`\frac{p^Tq}{\|p\|\|q\|}`{=tex} \]

### Skip-gram probability

\[ P(o\|c) = `\frac{e^{u_o^Tv_c}}`{=tex}
{`\sum`{=tex}\_{w`\in `{=tex}V}e^{u_w^Tv_c}} \]

### SVD

\[ X=U`\Sigma `{=tex}V\^T \]

### Negative sampling intuition

``` text
true pair -> score ↑
noise pair -> score ↓
```

公式忘一点没关系，但这个方向不能忘。

------------------------------------------------------------------------

# 14. Day 3 当天 GitHub 产物

你的 24 天计划规定：

``` text
notes/02_word_vectors.md
assignment-reflections/a1.md
```

建议结构：

``` text
cs224n-learning-guide/
├── notes/
│   ├── 00_numpy_for_dl.md
│   ├── 01_pytorch_core.md
│   └── 02_word_vectors.md
│
└── assignment-reflections/
    └── a1.md
```

------------------------------------------------------------------------

# 15. `notes/02_word_vectors.md` 应该写什么

不要抄 slides。

建议目录：

``` markdown
# Word Vectors

## 1. Why One-Hot Is Not Enough
## 2. Distributional Semantics
## 3. Context Windows
## 4. Count-Based Word Vectors
## 5. Co-occurrence Matrix
## 6. SVD and Dimensionality Reduction
## 7. Word2Vec Mental Model
## 8. Skip-gram vs CBOW
## 9. Softmax Objective
## 10. Negative Sampling
## 11. GloVe
## 12. Cosine Similarity
## 13. Intrinsic vs Extrinsic Evaluation
## 14. Analogies
## 15. Polysemy and Static Embedding Limitations
## 16. Synonym/Antonym Failure Cases
## 17. Bias in Word Embeddings
## 18. Shape Checklist
## 19. My Mental Model
```

------------------------------------------------------------------------

# 16. `assignment-reflections/a1.md` 应该写什么

不要公开 Stanford assignment 的完整 solution/code。

可以公开你的学习反思：

``` markdown
# Assignment 1 Reflection

## What I Implemented
- vocabulary extraction
- co-occurrence matrix
- truncated SVD
- embedding visualization

## Shape Flow
- M: [V,V]
- M_reduced: [V,k]
- norms: [V]
- normalized embeddings: [V,k]

## What Surprised Me
...

## Failure Cases
### Polysemy
...

### Synonym vs Antonym
...

### Analogy Failure
...

## Bias
...

## Bugs I Hit
...

## What I Would Test Next
...
```

------------------------------------------------------------------------

# 17. 今天不需要做的事情

为了保证 24 天主线不跑偏，今天不要额外展开：

-   从零训练大型 Word2Vec
-   手写高性能 negative sampling engine
-   深挖 hierarchical softmax
-   把 GloVe paper 每个公式全部推一遍
-   研究 subword embeddings / FastText
-   研究 contextual embeddings / BERT
-   大量调 gensim API
-   完整推导 Word2Vec gradients

这些不是"不重要"，而是现在的收益不如按计划推进。

Day 4 会进入：

``` text
Backprop
Matrix Calculus
Word2Vec derivatives
Adam
Dropout
```

------------------------------------------------------------------------

# 18. Day 3 最终验收清单

只有下面全部能打勾，才算完成 Day 3。

## Concepts

-   [ ] 我能解释 one-hot 的问题
-   [ ] 我能解释 distributional semantics
-   [ ] 我知道 context window 是什么
-   [ ] 我能解释 co-occurrence matrix
-   [ ] 我知道 `[V,V]` 为什么需要降维
-   [ ] 我能解释 SVD 在 A1 中的作用
-   [ ] 我能解释 Word2Vec 在预测什么
-   [ ] 我能区分 Skip-gram 与 CBOW
-   [ ] 我知道 center / outside 两套 vector
-   [ ] 我能写出 `P(o|c)` 的 softmax 结构
-   [ ] 我知道 naïve softmax 为什么贵
-   [ ] 我能解释 negative sampling
-   [ ] 我知道 GloVe 使用 global co-occurrence statistics
-   [ ] 我能写 cosine similarity
-   [ ] 我能区分 intrinsic / extrinsic evaluation
-   [ ] 我能解释 word analogy vector arithmetic
-   [ ] 我能解释 polysemy failure
-   [ ] 我能解释 synonym/antonym failure
-   [ ] 我能解释 embedding bias 的来源

## NumPy / Shape

-   [ ] 我能构造 `[V,V]` co-occurrence matrix
-   [ ] 我知道 `word2ind` 的作用
-   [ ] 我能解释 `[V,V] -> [V,k]`
-   [ ] 我能解释 `axis=1` row norm
-   [ ] 我能解释 `[V,k] / [V,1]` broadcasting
-   [ ] 我能从 `[V,D]` embedding matrix 取出 `[D]` word vector
-   [ ] 我能解释两个 `[D]` vector 如何得到 scalar cosine similarity

## Assignment

-   [ ] 我使用的是 2026 官网当前 A1
-   [ ] 我读过 README / starter notebook
-   [ ] Count-based 部分已完成或达到计划进度
-   [ ] SVD / plotting 已运行
-   [ ] GloVe 部分已运行
-   [ ] polysemy 分析已写
-   [ ] synonym / antonym failure 已写
-   [ ] analogy success / failure 已分析
-   [ ] bias 分析已完成
-   [ ] 没有搜索或复制公开 assignment solution

## GitHub

-   [ ] `notes/02_word_vectors.md`
-   [ ] `assignment-reflections/a1.md`
-   [ ] commit message 清楚
-   [ ] 没有公开 Stanford 作业完整答案 / starter solution

------------------------------------------------------------------------

# 19. 最后 5 分钟：口头自测

不看任何资料，直接回答：

### Q1

为什么：

``` text
hotel · motel = 0
```

并不意味着 hotel 和 motel 语义完全无关？

### Q2

给定：

``` text
U: [50000, 300]
v_c: [300]
```

`U @ v_c` 的 shape 是什么？每一个元素表示什么？

### Q3

为什么 naïve softmax 的 denominator 很贵？

### Q4

为什么 negative sampling 更便宜？

### Q5

给定：

``` text
M: [10000,10000]
```

做 truncated SVD 到 `k=100` 后，word embedding matrix 的 shape 是什么？

### Q6

为什么：

``` python
norms = np.linalg.norm(M, axis=1)
```

得到 `[V]`，而：

``` python
norms[:, None]
```

得到 `[V,1]`？

### Q7

为什么 antonym 可能在 embedding space 中非常接近？

### Q8

为什么 static embedding 很难表示 `bank` 的多个 meaning？

### Q9

intrinsic evaluation 很高，为什么不能直接证明 downstream system
一定更好？

### Q10

Word2Vec 与 GloVe 的共同目标是什么？训练信号又有什么不同？

如果 10 题中至少 8--9 题能完整讲出来，Day 3 的概念主线已经建立。

------------------------------------------------------------------------

# 20. Day 3 一句话总结

> **词向量的核心不是"把词变成一串数字"，而是利用语料中的上下文统计或预测信号，把语言中的部分语义关系编码进连续向量空间；而这个空间既有强大的规律，也有多义性、数据偏差、非线性关系和社会偏见等明确局限。**

------------------------------------------------------------------------

## 官方资料索引

-   CS224N Winter 2026：\
    https://web.stanford.edu/class/cs224n/

-   Stanford CS224N Spring 2024 Lecture 1 --- Intro and Word Vectors：\
    https://www.youtube.com/watch?v=DzpHeXVSC5I

-   Stanford CS224N Spring 2024 Lecture 2 --- Word Vectors and Language
    Models：\
    https://www.youtube.com/watch?v=nBor4jfWetQ

> 版本原则：**2026 slides / notes / assignments 为准；2024 Stanford
> 官方公开视频用于听课。**
