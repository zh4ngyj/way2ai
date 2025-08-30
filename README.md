# way2ai

## 阶段 0：学习准备（1–2 周）

学习目标
- 能够独立配置 Python 开发环境（Python、VSCode、Jupyter Notebook）
- 掌握 Git 的基本使用（clone、commit、push、branch）
- 能在 Linux 中进行基本操作（文件管理、进程查看、脚本执行）
- 能写并运行简单的 Python 程序

知识点
- 安装 **Python** 3.10
- 安装开发环境：**VSCode、Jupyter Notebook**
- **Git 基础**：版本控制、分支管理
- **Linux 基础**：文件操作、进程管理、常用命令（ls、cd、grep、top、htop）

实践任务
- 在 Jupyter Notebook 中运行 “Hello, World”
- 使用 GitHub 创建并提交一个项目仓库
- 在 Linux 中写一个 Shell 脚本批量重命名文件

## 阶段 1：数学基础（4–6 周）

学习目标
- 熟练掌握矩阵和向量运算，理解特征值、特征向量，能用奇异值分解 (SVD) 做降维（如 PCA）
- 理解常见概率分布（伯努利、二项、泊松、正态、指数等），能用极大似然估计 (MLE) 推断分布参数
- 能对多变量函数求导，理解反向传播与梯度下降
- 掌握常见优化方法（梯度下降、SGD、牛顿法、拉格朗日乘子法）
- 能够独立进行矩阵和向量的代数推导，理解奇异值分解（SVD）在降维和推荐系统中的应用。
- 掌握基本分布（正态、泊松、二项等）的性质和推导过程，能进行概率建模。
- 理解偏导数和梯度的概念，能够推导简单损失函数的梯度。
- 熟悉梯度下降的收敛原理，为机器学习优化打下基础。
- 能够读懂深度学习论文中常见的数学公式。

### 线性代数（Linear Algebra）

学习目标：掌握**矩阵和向量运算**，理解神经网络的线性运算本质

- **向量（Vector）**：加法、点积、范数
- **矩阵（Matrix）**：加法、乘法、转置、逆矩阵
- **矩阵的秩（Rank）**、**行列式（Determinant）**
- **特征值与特征向量（Eigenvalue & Eigenvector）**
- **奇异值分解**（Singular Value Decomposition, **SVD**）
- **正交矩阵、对角化、投影**

实践任务
- 用 NumPy 实现矩阵乘法
- 用奇异值分解 (SVD) 实现图像压缩
- 实现主成分分析 (Principal Component Analysis, PCA)

### 概率与统计 (Probability & Statistics)

- 概率基础：**条件概率**、**全概率公式**、**贝叶斯公式**
- 随机变量：**离散型**与**连续型**
- 常见分布：
    - 离散分布：**伯努利分布**、**二项分布**、**泊松分布**
    - 连续分布：**均匀分布**、**正态分布（高斯分布）**、**指数分布**、**卡方分布**、**t 分布**
- 数字特征：**期望**、**方差**、**协方差**、**相关系数**
- **大数定律**、**中心极限定理**
- 参数估计：**极大似然估计** (Maximum Likelihood Estimation, **MLE**)、**贝叶斯估计**
- 假设检验：**t 检验**、**卡方检验**、**p 值**

实践任务
- 用 Python 模拟抛硬币，验证二项分布收敛于正态分布
- 实现极大似然估计估算正态分布的均值和方差
- 从数据中计算相关系数并画散点图

### 微积分 (Calculus)

- 函数的**极限**与**连续性**
- **导数**与**偏导数**
- **多元函数**
- **梯度** (Gradient)、**方向导数**
- **链式法则** (Chain Rule)
- **泰勒展开** (Taylor Expansion)
- 积分（**定积分**、**不定积分**）
- **向量微积分**（主要学习梯度）

实践任务
- 用有限差分法实现数值微分
- 对 f(x,y)=x²+y² 使用梯度下降找到最小值
- 对神经网络的损失函数手动推导梯度

### 优化方法 (Optimization)
- **梯度下降** (Gradient Descent)
- **随机梯度下降** (Stochastic Gradient Descent, **SGD**)
- **动量法** (Momentum)、**AdaGrad**、**Adam**
- **牛顿法** (Newton’s Method)
- **拉格朗日乘子法** (Lagrange Multiplier Method)
- **凸优化**的概念

实践任务
- 用梯度下降最小化 f(x) = (x-3)²
- 用 Adam 优化 Logistic Regression
- 推导并实现支持向量机 (Support Vector Machine) 的对偶问题

### 离散数学
- 集合与映射。
- 图论（有向图、无向图、最短路径）。
- 逻辑与命题。
- 组合数学与排列组合。

## 阶段 2：编程与计算机基础（4–6 周）

学习目标
1. 熟练掌握 Python 编程，能写中等规模程序
2. 能使用 NumPy、Pandas、Matplotlib 进行数据处理和可视化
3. 理解常见数据结构和算法，并能手写实现
4. 能理解 CPU 与 GPU 的差别，能写多线程/多进程 Python 程序

### Python 编程
- 基础语法：变量、条件、循环
- 数据结构：列表、字典、集合、元组
- 函数、作用域、递归
- 类与对象、继承、多态
- 迭代器、生成器、装饰器
- 异常处理

实践任务
- 写一个学生成绩管理系统
- 写一个斐波那契数列生成器
- 写一个带缓存的函数装饰器

### 数据处理与可视化
- **NumPy（矩阵运算）**：矩阵运算、广播机制、随机数生成
- **Pandas（数据处理）**：DataFrame 创建、索引、分组、聚合
- **Matplotlib / Seaborn（数据可视化）**：折线图、柱状图、直方图、散点图、热力图

实践任务
- 用 Pandas 处理 Titanic 数据集，分析生存率
- 用 Matplotlib 画股票价格趋势图

### 算法与数据结构
- **时间复杂度**与**空间复杂度**
- 排序：**冒泡排序**、**快速排序**、**归并排序**
- 查找：**顺序查找**、**二分查找**、**哈希表**
- 数据结构：**栈**、**队列**、**链表**、**堆**
- 图算法：**深度优先搜索 (DFS)**、**广度优先搜索 (BFS)**、**最短路径 (Dijkstra, Floyd)**

实践任务
- 用 Python 实现快速排序
- 用 BFS 解决迷宫最短路径问题
- 用 Dijkstra 算法实现最短路径查询

### 计算机基础

- 操作系统：**进程与线程**、**内存管理**、**文件系统**
- 计算机体系结构：**CPU vs GPU**、**并行计算**
- 网络基础：**TCP/IP**、**HTTP 协议**

实践任务
- 在 Linux 中用 top 观察 CPU 占用率
- 写一个 Python 多线程爬虫
- 写一个简单的 HTTP 服务器

## 阶段 3：机器学习基础（6–8 周）

学习目标
- 理解并能实现基本机器学习算法（线性回归、逻辑回归、朴素贝叶斯、支持向量机、决策树、随机森林、KNN、K-Means）
- 掌握降维方法（PCA、LDA）及其数学原理
- 能使用交叉验证与多种指标评估模型（准确率、召回率、F1、ROC、AUC）
- 能识别过拟合与欠拟合，并使用正则化解决
- 能独立完成数据预处理、特征工程和模型训练。
- 能进行实验设计与结果分析，选择合适的模型。

### 核心算法
- **线性回归** (Linear Regression)
- **逻辑回归** (Logistic Regression)
- **朴素贝叶斯分类器** (Naive Bayes Classifier)
- **支持向量机** (Support Vector Machine, SVM)
- **决策树** (Decision Tree)、**随机森林** (Random Forest)
- **K 近邻算法** (K-Nearest Neighbors, KNN)
- **聚类算法**：**K 均值** (K-Means)、**层次聚类**
- **降维**：**主成分分析** (Principal Component Analysis, PCA)、**线性判别分析** (Linear Discriminant Analysis, LDA)

### 模型评估
- 损失函数：均方误差 (Mean Squared Error)、交叉熵 (Cross-Entropy)
- 模型评估指标：准确率、精确率、召回率、F1 分数、ROC 曲线、AUC
- 交叉验证 (Cross Validation)、过拟合与正则化 (L1/Lasso, L2/Ridge)

实践任务
- 用 Scikit-Learn 实现线性回归与逻辑回归
- 在 MNIST 手写数字数据集上用 SVM 分类
- 在 Titanic 数据集上训练决策树并可视化

## 阶段 4：深度学习核心（8–10 周）

学习目标
- 掌握神经网络的构成（感知机、激活函数、损失函数、反向传播）
- 能构建并训练 CNN、RNN、LSTM 网络
- 理解并使用常见训练技巧（Batch Normalization、Dropout、学习率调度）
- 能熟练使用 PyTorch/TensorFlow 构建深度学习模型
- 理解神经网络的数学推导与反向传播机制
- 能够训练图像分类、文本分类等基础深度学习任务
- 能够解释模型表现，并进行超参数调优

### 神经网络基础
- **感知机** (Perceptron)
- **多层感知机** (Multilayer Perceptron, MLP)
- **激活函数**：Sigmoid、Tanh、ReLU、Leaky ReLU、Softmax
- **损失函数**：均方误差、交叉熵损失
- **反向传播算法** (Backpropagation)

### 常见网络
- **卷积神经网络** (Convolutional Neural Network, CNN)
- **循环神经网络** (Recurrent Neural Network, RNN)
- **长短期记忆网络** (Long Short-Term Memory, LSTM)
- **门控循环单元** (Gated Recurrent Unit, GRU)
- **Seq2Seq 模型**

### 优化技巧
- **Batch Normalization**
- **Dropout**
- **学习率调度** (Learning Rate Scheduler)

### 深度学习框架
- PyTorch / TensorFlow 基础
- 模型搭建、训练与调试
- GPU 加速

实践任务
- 用 PyTorch 从零实现一个全连接神经网络
- 训练 CNN 在 CIFAR-10 数据集上分类
- 用 RNN 生成文本

## 阶段 5：大语言模型（LLM）进阶（10–12 周）

学习目标
- 深入理解 Transformer 架构与自注意力机制
- 理解预训练语言模型（BERT 的掩码语言模型，GPT 的因果语言模型）
- 能够使用 Hugging Face Transformers 微调预训练模型
- 掌握 LLM 优化与压缩方法（LoRA、知识蒸馏、量化）

### 传统NLP
- 文本表示：**Bag of Words**、**TF-IDF**。
- **Word2Vec**、**GloVe**。
- 语言模型（**n-gram**）

### 深度学习 NLP
- RNN/LSTM 在 NLP 中的应用。
- Seq2Seq 模型与注意力机制（Attention）

### Transformer 架构
- **自注意力机制** (Self-Attention)
- **多头注意力机制** (Multi-Head Attention)
- **位置编码** (Positional Encoding)
- **编码器–解码器**结构 (Encoder–Decoder)
- **预训练任务**：掩码语言模型 (Masked Language Modeling, **BERT**)、因果语言模型 (Causal LM, **GPT**)、T5模型（Transfer Text-to-Text Transformer **T5**）

### LLM 技术
- 微调方法：**全参数微调** (Full Fine-Tuning)、**低秩适配** (Low-Rank Adaptation, LoRA)
- **参数高效微调** (PEFT)
- **知识蒸馏** (Knowledge Distillation)
- **模型压缩与量化**

实践任务
- 用 PyTorch 实现一个 Mini-Transformer
- 用 Hugging Face Transformers 微调 GPT-2 做文本生成
- 用 BERT 做文本分类（情感分析）
- 使用 Hugging Face Transformers
- 加载预训练模型（BERT、GPT-2、LLaMA）
- 实现文本生成、问答、摘要等应用

## 阶段 6：进阶与科研 / 实战（长期）

学习目标
- 掌握前沿 LLM 技术（RLHF、RAG、MoE、多模态）
- 理解分布式训练与推理优化方法
- 能将模型部署到实际应用（API、Web 服务）
- 具备科研能力，能复现论文并提出改进

前沿研究
- **基于人类反馈的强化学习** (Reinforcement Learning with Human Feedback, **RLHF**)
- **检索增强生成** (Retrieval-Augmented Generation, **RAG**)
- **混合专家模型** (Mixture of Experts, **MoE**)
- **多模态大模型**（视觉 + 文本）
- ...

系统与工程
- **分布式训练** (Data Parallel, Model Parallel, ZeRO)
- **推理加速**：张量并行、流水线并行、KV 缓存
- **模型部署**：FastAPI、Docker、Kubernetes、TensorRT
- **MLOps**：模型版本管理、监控、持续训练

实践任务
- 复现 BERT 或 GPT 的论文实验
- 部署一个基于 RAG 的问答系统
- 用 LoRA 在小型 GPU 上微调大模型