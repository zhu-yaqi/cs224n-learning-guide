# CS224N 24 天高强度学习计划：从 NumPy 到可写进简历的 NLP/LLM 项目

> 适合背景：已经学过机器学习，但 NumPy 基础薄弱；目标是尽快、扎实完成 CS224N，并衔接 CS336；同时把学习过程整理成 GitHub 教程，并把大型任务沉淀成实习项目经验。
>
> 建议强度：每天约 5.5–6.5 小时，共约 130–145 小时。
>
> 版本策略：**以 Stanford CS224N Winter 2026 官网的 slides / assignments / project requirements 为准；公开视频使用 Stanford Online 的 Spring 2024 官方 YouTube playlist。不要混做旧版作业。**

---

## 1. 先说结论：建议 24 天

如果目标只是“看完课”，14–18 天可以完成；但如果目标是：

- 真正会 NumPy / PyTorch；
- 独立完成 2026 版 4 个作业；
- 能把 Transformer、LLM eval、RAG/PEFT 等内容讲明白；
- 做出一个可展示的项目；
- 整理成 GitHub 教程；
- 并为 CS336 留出不重复造轮子的空间；

那么我建议 **24 天**。

节奏建议：

- **每天 5.5–6.5 小时**：24 天。
- 每天只有 3–4 小时：建议拉长到 35–40 天。
- 真正全职冲刺、每天 8 小时以上：可以压到 20–21 天，但项目和复盘会比较赶。

### 每天固定节奏

1. 20–30 min：先扫 slides / handout，明确今天的问题。
2. 60–100 min：看对应 lecture；核心数学 1.25×，熟悉/重复部分 1.5–1.75×。
3. 30–45 min：闭卷写一页 notes：公式、shape、为什么。
4. 2.5–3.5 h：代码 / assignment / mini-lab。
5. 30–45 min：整理 GitHub 学习笔记、记录 bug、commit。

原则：**看课不是进度，完成代码、推导和复盘才算进度。**

---

## 2. 资源总入口

### CS224N

- Winter 2026 官方课程页  
  https://web.stanford.edu/class/cs224n/
- Stanford 官方公开 Spring 2024 视频 playlist  
  https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D
- 当前作业和 slides：从 2026 官网 Schedule 中的 `code / handout / slides / notes` 进入。
- Assignment 2 handout  
  https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf
- Assignment 3 handout  
  https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf
- Assignment 4 handout  
  https://web.stanford.edu/class/cs224n/assignments_w26/a4.pdf
- Default Final Project（GPT-2）handout  
  https://web.stanford.edu/class/cs224n/project/DFP_Instructions.pdf

> 注意：搜索引擎很容易搜到 archived CS224N。**作业只从 2026 主课程页进入。**

### NumPy / Python

- NumPy 官方 Quickstart  
  https://numpy.org/doc/stable/user/quickstart
- Stanford CS231N Python/NumPy Tutorial  
  https://cs231n.github.io/python-numpy-tutorial/
- CS224N 2026 官网 Week 1 的 Python Review Session：直接点课程页的 `slides` 和 `colab`。

### PyTorch

- PyTorch Learn the Basics  
  https://docs.pytorch.org/tutorials/beginner/basics/
- PyTorch Autograd  
  https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
- CS224N 2026 官网的 PyTorch Tutorial Session Colab。

### Hugging Face

- Hugging Face LLM/NLP Course  
  https://huggingface.co/docs/course/chapter1/1
- Transformers fine-tuning docs  
  https://huggingface.co/docs/transformers/main/training

### 后续 CS336

- Stanford CS336: Language Modeling from Scratch  
  https://cs336.stanford.edu/

CS336 会大量覆盖：tokenizer、Transformer LM、optimizer、训练、GPU/Triton、FlashAttention、并行、scaling laws、pretraining data、post-training / RL。  
所以 CS224N 阶段不要花大量时间重复造这些“系统级轮子”。

---

## 3. 2026 CS224N 作业重点

### Assignment 1 — Word Vectors

目标：建立 embedding / similarity / vector semantics 的直觉，并练 NumPy。

建议：**完整做，但限时 1 天左右。**

重点掌握：

- ndarray / shape / axis；
- slicing / boolean indexing；
- broadcasting；
- matrix multiplication；
- cosine similarity；
- co-occurrence；
- SVD / dimensionality reduction；
- GloVe / embedding 的局限。

### Assignment 2 — Word2Vec and Dependency Parsing

官方内容：

- Word2Vec 数学和梯度；
- Adam、Dropout；
- transition-based dependency parsing；
- 用 PyTorch 实现并训练 neural dependency parser。

建议：**完整做。**

它非常适合检查你是否真的掌握：

- vectorized derivatives；
- tensor shape；
- PyTorch forward；
- training loop；
- embedding lookup；
- debugging；
- evaluation。

### Assignment 3 — Self-Attention and Transformers

官方内容：

- self-attention 直觉；
- positional encoding；
- 几乎从头实现 Transformer；
- 训练并做若干加速 / 改进实验。

建议：**这是整个 CS224N 最该认真做的作业之一。完整做，自己画 shape table。**

但不要在这里额外扩展成“完整 GPT pretraining framework”，因为 CS336 Assignment 1 会更系统地做 tokenizer、Transformer、optimizer 和 LM training。

### Assignment 4 — LLM Evals

当前课程的 A4 聚焦：

- standard closed-ended benchmarking；
- LLM-as-a-judge；
- red teaming / failure modes；
- LLM evaluation 的设计和局限。

建议：**完整做，而且把它升级成一个独立 evaluation harness。**

这个方向和实习项目非常搭，而且不像“从头训练 Transformer”那样与 CS336 高度重复。

---

## 4. CS224N 与 CS336 的投入策略

| 内容 | CS224N 投入 | 原因 |
|---|---:|---|
| NumPy / tensor shape | 高 | 这是后面所有实现的基本功 |
| PyTorch / autograd / training loop | 高 | CS336 明确需要很强的 PyTorch |
| Word vectors | 中 | 建立 NLP 表示学习直觉，做完 A1 即可 |
| Dependency parsing | 中 | A2 做完即可，不额外深挖传统 parsing |
| RNN / LSTM / seq2seq | 低-中 | 理解 Transformer 为什么出现，不做大型旧式 NMT 项目 |
| Transformer / attention | **最高** | A3 必须彻底掌握 |
| Tokenization | 低 | CS336 A1 会从头实现 tokenizer |
| Pretraining / optimizer internals | 中低 | 概念理解；实现深度留给 CS336 |
| GPU / kernels / distributed systems | 很低 | 留给 CS336 A2 |
| Scaling laws / training data | 低-中 | 留给 CS336 A3/A4 深挖 |
| SFT / RLHF / DPO | 中 | CS224N 学概念，CS336 A5 再实现 |
| Prompting / PEFT / LoRA | 高 | 实习常用，和 CS336 主线互补 |
| RAG / agents / tool use | 高 | 实用且项目化价值高 |
| Benchmark / LLM eval | **最高** | A4 + 项目可直接形成作品 |
| Default GPT-2 final project | 不建议作为主项目 | 与 CS336 A1 的 from-scratch LM 高度重叠 |
| Custom final project | **高** | 更适合做简历项目 |

---

# 5. 24 天逐日计划

## Day 1 — NumPy 最低必要基础

**时间：6 h**

学习：

- CS231N Python/NumPy Tutorial；
- NumPy Quickstart；
- CS224N Python Review slides / Colab。

必须会：

- `np.array`
- `shape`, `ndim`, `dtype`
- `reshape`, `transpose`, `squeeze`
- slicing / integer indexing / boolean indexing
- `sum/mean/max/argmax` + `axis`
- broadcasting
- `@`, `dot`
- `stack/concatenate`
- `np.linalg.svd`
- vectorization

自己做一个 `numpy_bootcamp.ipynb`：

1. 不用 for-loop 实现 pairwise cosine similarity；
2. stable softmax；
3. cross entropy；
4. mini batch matrix operation；
5. SVD 降维并画 2D points。

**当天 GitHub 产物**

`notes/00_numpy_for_dl.md`

重点写“shape 如何流动”，不要抄 API 手册。

---

## Day 2 — PyTorch 必备能力

**时间：6 h**

学习：

- PyTorch Learn the Basics：
  - Tensors
  - Build Model
  - Autograd
  - Optimization
- CS224N PyTorch Tutorial Colab。

做：

- 把 Day 1 的 softmax / cross entropy 用 torch 改写；
- 手算一个小网络梯度，再用 autograd 验证；
- 自己写最小 training loop：
  - zero grad
  - forward
  - loss
  - backward
  - optimizer.step

重点掌握：

- tensor vs numpy；
- CPU/GPU；
- `requires_grad`；
- `.backward()`；
- `nn.Module`；
- parameters；
- train/eval mode；
- `no_grad()`；
- broadcasting；
- batch dimension。

**产物**

`notes/01_pytorch_core.md`  
`labs/autograd_sanity_checks.ipynb`

---

## Day 3 — Word vectors + Assignment 1

**时间：5.5–6 h**

听：

- 2024 Lecture 1 — Word Vectors
- 2024 Lecture 2 — Word Vectors and Language Models

辅助：

- 2026 Word Vectors slides / notes；
- 2026 History of NLP slides快速扫一遍即可。

做：

- 2026 Assignment 1；
- 把每一个 NumPy 操作的输入输出 shape 写在旁边；
- 理解 co-occurrence、SVD、GloVe、cosine similarity；
- 做 embedding failure cases / bias / polysemy 的简短总结。

**产物**

`notes/02_word_vectors.md`  
`assignment-reflections/a1.md`

---

## Day 4 — Backprop / matrix calculus + A2 理论部分

**时间：6 h**

听：

- 2024 Lecture 3 — Backpropagation and Neural Network Basics
- 对照 2026 Backpropagation and Neural Network Basics slides / notes。

做：

- A2 Word2Vec derivatives；
- A2 Adam / Dropout；
- 用 finite difference 检查至少一个你推导的梯度；
- 写“向量化梯度 shape checklist”。

重点：

- softmax + CE 的梯度；
- chain rule；
- matrix derivative；
- Jacobian 不要死背，用 shape + differential 理解；
- Adam intuition；
- dropout train vs eval。

**产物**

`notes/03_backprop_shape_first.md`

---

## Day 5 — Dependency parsing + A2 coding

**时间：6.5 h**

听：

- 2024 Lecture 4 — Dependency Parsing。

做：

- transition mechanics；
- minibatch parsing；
- parser model；
- training loop；
- 先 debug set，再 full set；
- 记录 dev UAS；
- 挑 5 个错误 parse 做 error analysis。

重点不是 parsing 本身，而是：

- batching；
- embedding lookup；
- vectorization；
- PyTorch model wiring；
- debug workflow。

**产物**

`assignment-reflections/a2.md`

---

## Day 6 — A2 收尾 + PyTorch Debugging Day

**时间：5 h**

做：

- A2 全部测试；
- 重新从空白纸画一次 forward graph；
- 写 5 个 PyTorch 常见 bug：
  - shape mismatch
  - wrong device
  - forgotten zero_grad
  - train/eval mode
  - accidental Python loop

再写一个 100 行以内的小 classifier，确保不依赖 starter code。

**产物**

`notes/04_debugging_pytorch_models.md`

---

## Day 7 — RNN / LSTM / Seq2Seq：只学历史桥梁

**时间：4.5–5 h**

听：

- 2024 Lecture 5 — Recurrent Neural Networks；
- 2024 Lecture 6 — Sequence to Sequence Models。

速度可以 1.5×。

目标只到：

- LM factorization；
- RNN hidden state；
- vanishing/exploding gradient；
- LSTM 为什么出现；
- encoder-decoder；
- attention 为什么能解决固定 bottleneck。

**不要做旧版 NMT Assignment。**

写一篇：

`Why Transformers replaced RNNs for modern LMs`

**产物**

`notes/05_rnn_to_transformer.md`

---

## Day 8 — Transformer 核心课：这一天必须慢

**时间：6 h**

听：

- 2024 Transformers lecture；
- 2026 Transformers slides / notes。

读：

- *Attention Is All You Need*：重点看 architecture、scaled dot-product attention、multi-head attention、positional encoding。

手推：

- Q/K/V dimensions；
- `QK^T / sqrt(d_k)`；
- causal mask；
- softmax axis；
- multi-head reshape；
- residual；
- LayerNorm；
- parameter count 粗估；
- attention complexity。

要求：

**不看代码，自己在纸上从 `[B,T,D]` 推到 attention 输出 `[B,T,D]`。**

**产物**

`notes/06_transformer_shapes.md`

---

## Day 9 — Assignment 3 前半

**时间：6.5–7 h**

做：

- A3 self-attention questions；
- positional encoding；
- Transformer implementation；
- 写单元测试：
  - output shape
  - masking
  - permutation behavior
  - deterministic seed

每一个模块写 docstring：

- input shape；
- output shape；
- mask shape；
- 复杂度。

**产物**

`assignment-reflections/a3_part1.md`

---

## Day 10 — Assignment 3 后半 + 实验

**时间：6.5–7 h**

做：

- 训练 baseline；
- 按 handout 做至少 3 个不同思路的改动；
- 画 learning curves；
- 记录 100 steps 后的 loss；
- 写 ablation table。

把作业中的实验抽象成一篇：

`How I debugged and sped up a tiny Transformer`

**注意：**

到这里就停。  
不要扩成完整 tokenizer + GPT pretraining engine，后面 CS336 A1 会更系统地做。

---

## Day 11 — Pretraining：理解，不深挖实现

**时间：4.5–5 h**

看：

- 2024 Pretraining lecture；
- 2026 Pretraining (Scaling, Systems, Data) slides。

了解：

- causal LM vs masked LM；
- BERT / GPT；
- data → tokenize → batch → train；
- scaling 的基本思想；
- compute / memory / data 之间关系。

对下面内容只记问题，不实现：

- tokenizer from scratch；
- optimizer from scratch；
- efficient attention；
- GPU kernels；
- distributed training；
- scaling law fitting。

**这些留给 CS336。**

**产物**

`notes/07_pretraining_map.md`

---

## Day 12 — Post-training + Final Project 选题

**时间：5.5 h**

看：

- 2024 Post-training；
- 2026 Post-training (RLHF, SFT, DPO) slides；
- 2026 Final Projects / Practical Tips slides。

掌握：

- SFT；
- preference data；
- reward model；
- RLHF pipeline；
- DPO objective 的直觉。

今天确定你的 custom project：

## 推荐项目题目

**DomainAdapt-Eval: RAG vs LoRA for Domain QA**

研究问题：

> 在领域问答的 domain shift 下，Base / RAG / LoRA / RAG+LoRA 各自在什么条件下更有效？

比较：

1. Base model
2. Base + RAG
3. LoRA fine-tuned
4. LoRA + RAG

评价：

- task metric；
- LLM-as-judge；
- latency；
- retrieval hit rate；
- failure taxonomy。

这样正好把后面的：

- PEFT
- RAG
- evaluation
- error analysis

整合成一个简历项目，而且不会和 CS336 的 from-scratch LM 主线重复。

**产物**

`project/proposal.md`

---

## Day 13 — Efficient Adaptation / PEFT / LoRA

**时间：6 h**

看：

- 2026 Efficient Adaptation (Prompting + PEFT) slides；
- LoRA paper 的方法与实验核心；
- Hugging Face fine-tuning / PEFT 相关教程。

mini-lab：

- 选 0.5B–1.5B 量级开源模型；
- 跑通 inference；
- 跑通 LoRA fine-tuning 小数据；
- 比较 trainable parameter count；
- 保存 adapter。

不要追求分数，目标是 pipeline 全通。

**产物**

`labs/lora_minilab/`  
`notes/08_peft_lora.md`

---

## Day 14 — Agents / Tool Use / RAG

**时间：6 h**

看：

- 2026 Agents, Tool Use, and RAG slides；
- ReAct / RAG 论文看核心结构，不做逐段精读。

mini-lab：

- corpus ingest；
- chunking；
- embeddings；
- vector retrieval；
- top-k；
- prompt composition；
- generation。

至少做两个 ablation：

- chunk size；
- top-k。

**产物**

`labs/rag_minilab/`  
`notes/09_rag_failure_modes.md`

---

## Day 15 — Benchmarking / Evaluation + A4 前半

**时间：6 h**

听：

- 2024 Lecture 11 — Benchmarking；
- 2026 Benchmarking and Evaluation slides。

开始 A4：

- closed-ended benchmark；
- evaluator design；
- LLM-as-judge。

你要学会区分：

- capability；
- correctness；
- preference；
- safety；
- robustness；
- latency / cost。

**产物**

`notes/10_llm_evaluation.md`

---

## Day 16 — Assignment 4 完成 + Eval Harness

**时间：6 h**

做：

- 完成 A4；
- 做 red-team / failure exploration；
- 把作业思路抽象成你自己的 evaluation harness：
  - dataset loader
  - model adapter
  - evaluator
  - metrics
  - result JSON
  - summary table

这个 harness 之后直接用于 custom project。

**产物**

`labs/eval_harness/`  
`assignment-reflections/a4.md`

---

## Day 17 — Reasoning + Tokenization：有选择地学

**时间：5 h**

看：

- 2026 Reasoning 1；
- 2026 Reasoning 2；
- 2026 Tokenization and Multilinguality guest lecture slides。

如果需要公开视频补充：

- 2024 Reasoning and Agents 对应 lecture。

Reasoning 重点：

- CoT；
- self-consistency；
- verifier；
- test-time compute；
- reasoning evaluation。

Tokenization 只理解：

- BPE / subword 的动机；
- fertility；
- multilingual inequity；
- tokenization 对 compute 和 evaluation 的影响。

**不要今天自己实现 BPE。CS336 A1 会做。**

**产物**

`notes/11_reasoning_and_tokenization.md`

---

## Day 18 — 选择性现代主题 + 总复盘

**时间：4–5 h**

从 2026 课程中选择性过：

- Interpretability；
- Social and Broader Impacts；
- Multimodality；
- Open Questions in NLP。

策略：

- slides 为主；
- 对你未来方向特别相关的再精听 / 读 paper。

然后做一次课程闭卷复盘：

回答：

1. Word2Vec 在优化什么？
2. autograd 真正在做什么？
3. Transformer 每一步 shape 是什么？
4. causal mask 为什么必要？
5. pretraining / SFT / DPO 分别解决什么？
6. RAG 与 LoRA 的 failure mode 有何不同？
7. 如何证明一个 LLM improvement 不是 benchmark artifact？

**产物**

`notes/12_cs224n_master_summary.md`

---

# Day 19–24：把课程变成真正项目

## Day 19 — Project baseline + dataset

**时间：6 h**

完成：

- 明确定义 task / metric；
- 清洗数据；
- train/dev/test split；
- base model inference；
- baseline results；
- experiment config；
- reproducible seed。

必须先有 baseline，再加复杂方法。

---

## Day 20 — RAG system

**时间：6.5 h**

完成：

- index；
- retrieval；
- generation；
- retrieval metrics；
- top-k / chunk size ablation；
- 20 个 failure cases。

记录：

- retrieval failure；
- answer synthesis failure；
- hallucination despite correct retrieval。

---

## Day 21 — LoRA system

**时间：6.5–7 h**

完成：

- training data format；
- LoRA training；
- checkpoint / adapter；
- validation；
- base vs LoRA；
- 参数量 / 显存 / 时间统计。

至少做一个：

- LoRA rank ablation；
- training data size ablation。

---

## Day 22 — RAG + LoRA + ablation matrix

**时间：6.5 h**

跑四组：

| System | Retrieval | Fine-tuning |
|---|---|---|
| Base | No | No |
| RAG | Yes | No |
| LoRA | No | Yes |
| RAG + LoRA | Yes | Yes |

形成统一结果表。

额外记录：

- latency；
- token usage；
- model size / adapter size；
- retrieval hit；
- judge score；
- task score。

---

## Day 23 — Evaluation / Error Analysis / Report

**时间：6 h**

做：

- bootstrap / variance（有余力再做）；
- error taxonomy；
- qualitative examples；
- failure analysis；
- limitations；
- 哪种方法在什么场景更适合。

写：

`project/report.md`

结构：

1. Problem
2. Motivation
3. Dataset
4. Methods
5. Experimental Setup
6. Results
7. Ablations
8. Error Analysis
9. Limitations
10. Reproduction

---

## Day 24 — GitHub 教程化 + 简历化 + CS336 交接

**时间：6 h**

完成 public README：

- What this repo teaches
- 24-day roadmap
- Environment
- Concept map
- Notes index
- Mini-labs
- Project
- Results
- Reproduction
- What to learn next: CS336

项目 README 加：

- 架构图；
- 实验表；
- failure cases；
- 复现命令；
- demo GIF / screenshot（可选）。

准备简历 bullet，但**只有测出真实结果后才填数字**：

> Built a reproducible domain-QA evaluation pipeline comparing base, RAG, LoRA, and RAG+LoRA systems; designed retrieval and LLM-judge evaluations, ran ablations on retrieval/fine-tuning hyperparameters, and analyzed failure modes across ___ examples, improving ___ by ___ while measuring latency/cost tradeoffs.

最后做 CS336 readiness checklist：

- [ ] 能熟练处理 `[B,T,D]` tensor shapes
- [ ] 能独立写 PyTorch training loop
- [ ] 能解释 autograd / optimizer
- [ ] 能从公式写 self-attention
- [ ] 能 debug masking / broadcasting
- [ ] 能读 profiler / 显存错误的基本信息
- [ ] 不依赖 Hugging Face 才能理解 Transformer 内部
- [ ] 明白 tokenization / pretraining / eval 的全流程

然后开始 CS336。

---

## 6. 作业参考应该怎么看

CS224N 官方明确要求学生不要去网上找 assignment solutions / implementations。

因此，你的“作业参考”建议只用：

1. 当年官方 handout；
2. 当年 lecture slides / notes；
3. NumPy / PyTorch 官方文档；
4. handout 给出的论文；
5. CS231N 的 matrix calculus / backprop notes；
6. 自己写的小规模 sanity check；
7. 只问“概念、API、debug 方法”，不要让 AI 或搜索结果直接给完整作业答案。

推荐 debug 顺序：

1. 先检查 shape；
2. 再检查数值范围；
3. 再检查 mask；
4. 再检查 gradient；
5. 再检查 optimizer；
6. 最后才怀疑模型设计。

---

## 7. GitHub 怎么做，既能教后来人又不变成答案仓库

建议分成两个 repo。

### Private repo：真正作业

`cs224n-coursework-private`

放：

- Stanford starter code；
- 你填写后的 assignment code；
- written solutions；
- raw experiment logs。

保持 private。

### Public repo：教程 + 自己的项目

`cs224n-learning-guide`

建议结构：

```text
cs224n-learning-guide/
├── README.md
├── roadmap.md
├── notes/
│   ├── 00_numpy_for_dl.md
│   ├── 01_pytorch_core.md
│   ├── 02_word_vectors.md
│   ├── 03_backprop_shape_first.md
│   ├── 05_rnn_to_transformer.md
│   ├── 06_transformer_shapes.md
│   ├── 07_pretraining_map.md
│   ├── 08_peft_lora.md
│   ├── 09_rag_failure_modes.md
│   ├── 10_llm_evaluation.md
│   ├── 11_reasoning_and_tokenization.md
│   └── 12_cs224n_master_summary.md
├── labs/
│   ├── numpy_bootcamp/
│   ├── autograd_sanity_checks/
│   ├── tiny_attention_from_scratch/
│   ├── lora_minilab/
│   ├── rag_minilab/
│   └── eval_harness/
├── assignment-reflections/
│   ├── a1.md
│   ├── a2.md
│   ├── a3.md
│   └── a4.md
├── project/
│   ├── README.md
│   ├── proposal.md
│   ├── configs/
│   ├── src/
│   ├── eval/
│   ├── results/
│   └── report.md
└── resources.md
```

`assignment-reflections/` 可以公开：

- 你学到了什么；
- 哪类 bug 最难；
- 哪个公式容易错；
- 实验曲线；
- 错误分析；
- 你自己的 toy reimplementation。

不要公开：

- 官方 TODO 的完整填空答案；
- 可直接提交 Gradescope 的完整代码；
- written assignment 的逐题标准答案。

这样 repo 反而更像真正的教程，而不是答案库。

---

## 8. 笔记模板

每个 topic 只写下面 7 部分：

```md
# Topic

## 1. Problem
它解决什么问题？

## 2. Core idea
一句话解释。

## 3. Math
最重要的公式。

## 4. Shapes
每个 tensor 的 shape。

## 5. Minimal implementation
自己写的 toy code，不复制课程答案。

## 6. Failure modes / common bugs
哪里最容易错？

## 7. Connection
它与 Transformer / LLM / CS336 有什么关系？
```

这样你最后的 GitHub 会非常整洁。

---

## 9. 这套计划完成后的目标状态

完成 24 天后，你应该能够：

- 用 NumPy 做向量化计算，不依赖 Python for-loop；
- 熟练处理 PyTorch tensor shapes / broadcasting；
- 自己写 training loop 和 debug；
- 解释 Word2Vec / embedding；
- 理解 RNN → attention → Transformer 的演化；
- 从公式写一个 tiny self-attention / Transformer；
- 解释 pretraining / SFT / DPO；
- 搭建 LoRA / RAG；
- 设计 LLM eval，而不是只看“模型感觉不错”；
- 做实验、画表、做 ablation、写 error analysis；
- 在 GitHub 上展示一套完整学习教程；
- 有一个可以写进 NLP / LLM / Applied ML 实习简历的项目；
- 进入 CS336 时，把时间集中在 tokenizer、from-scratch LM、GPU systems、FlashAttention、distributed training、scaling、data 和 RL，而不是重复 CS224N 的应用层内容。

---

## 10. 最重要的三条

1. **A3 Transformer 和 A4 Evaluation 一定认真做。**
2. **默认 GPT-2 final project 不作为你的主项目；改做 custom project，把 RAG + PEFT + evaluation 做深。**
3. **公开 GitHub 做教程，不做答案库；作业代码 private，自己的 labs/project public。**

这会让 CS224N 不只是“上完一门课”，而是成为你进入 CS336 和找 NLP/LLM 实习的第一块完整作品。
