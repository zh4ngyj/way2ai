# way2ai

## Stage 0: Learning Preparation (1–2 Weeks)

Learning Goals
- Be able to independently set up a Python development environment (Python, VSCode, Jupyter Notebook)
- Master the basics of Git (clone, commit, push, branch)
- Perform basic operations in Linux (file management, process monitoring, script execution)
- Write and run simple Python programs

Key Knowledge
- Install **Python** 3.10
- Set up development environment: **VSCode, Jupyter Notebook**
- **Git Basics**: version control, branch management
- **Linux Basics**: file operations, process management, common commands (ls, cd, grep, top, htop)

Practical Tasks
- Run “Hello, World” in Jupyter Notebook
- Create and commit a project repository on GitHub
- Write a shell script in Linux to batch rename files

## Stage 1: Mathematical Foundations (4–6 Weeks)

Learning Goals
- Master matrix and vector operations, understand eigenvalues and eigenvectors, and use Singular Value Decomposition (SVD) for dimensionality reduction (e.g., PCA)
- Understand common probability distributions (Bernoulli, Binomial, Poisson, Normal, Exponential, etc.) and estimate parameters using Maximum Likelihood Estimation (MLE)
- Compute derivatives for multivariable functions, understand backpropagation and gradient descent
- Master common optimization methods (Gradient Descent, SGD, Newton's Method, Lagrange Multipliers)
- Perform algebraic derivations with matrices and vectors; understand SVD applications in dimensionality reduction and recommendation systems
- Understand properties of basic distributions (Normal, Poisson, Binomial) and perform probability modeling
- Grasp partial derivatives and gradients; derive gradients for simple loss functions
- Understand convergence principles of gradient descent for machine learning optimization
- Be able to read and understand common mathematical formulas in deep learning papers

### Linear Algebra

Learning Goals: Master **matrix and vector operations** and understand the linear operations in neural networks

- **Vector**: addition, dot product, norm
- **Matrix**: addition, multiplication, transpose, inverse
- **Matrix rank** and **determinant**
- **Eigenvalues & Eigenvectors**
- **Singular Value Decomposition (SVD)**
- **Orthogonal matrices, diagonalization, projection**

Practical Tasks
- Implement matrix multiplication using NumPy
- Compress images using SVD
- Implement Principal Component Analysis (PCA)

### Probability & Statistics

- Probability basics: **Conditional Probability**, **Law of Total Probability**, **Bayes’ Theorem**
- Random variables: **Discrete** and **Continuous**
- Common distributions:
    - Discrete: **Bernoulli**, **Binomial**, **Poisson**
    - Continuous: **Uniform**, **Normal (Gaussian)**, **Exponential**, **Chi-square**, **t-distribution**
- Numerical characteristics: **Expectation**, **Variance**, **Covariance**, **Correlation coefficient**
- **Law of Large Numbers**, **Central Limit Theorem**
- Parameter estimation: **Maximum Likelihood Estimation (MLE)**, **Bayesian Estimation**
- Hypothesis testing: **t-test**, **Chi-square test**, **p-value**

Practical Tasks
- Simulate coin tosses in Python and verify Binomial distribution converges to Normal distribution
- Estimate mean and variance of a normal distribution using MLE
- Compute correlation coefficients from data and plot scatter plots

### Calculus

- Function **limits** and **continuity**
- **Derivatives** and **partial derivatives**
- **Multivariable functions**
- **Gradient** and **directional derivative**
- **Chain rule**
- **Taylor expansion**
- Integration (**definite** and **indefinite**)
- **Vector calculus** (focus on gradients)

Practical Tasks
- Implement numerical differentiation using finite differences
- Use gradient descent to find the minimum of f(x,y)=x²+y²
- Manually derive gradients for a neural network loss function

### Optimization

- **Gradient Descent**
- **Stochastic Gradient Descent (SGD)**
- **Momentum**, **AdaGrad**, **Adam**
- **Newton’s Method**
- **Lagrange Multiplier Method**
- Concept of **Convex Optimization**

Practical Tasks
- Minimize f(x) = (x-3)² using gradient descent
- Optimize Logistic Regression using Adam
- Derive and implement the dual problem for Support Vector Machine (SVM)

### Discrete Mathematics

- Sets and mappings
- Graph theory (directed, undirected, shortest path)
- Logic and propositions
- Combinatorics and permutations

## Stage 2: Programming & Computer Fundamentals (4–6 Weeks)

Learning Goals
1. Master Python programming and write medium-scale programs
2. Use NumPy, Pandas, Matplotlib for data processing and visualization
3. Understand common data structures and algorithms and implement them by hand
4. Understand the differences between CPU and GPU; write multithreaded/multiprocess Python programs

### Python Programming

- Basic syntax: variables, conditions, loops
- Data structures: lists, dictionaries, sets, tuples
- Functions, scopes, recursion
- Classes & objects, inheritance, polymorphism
- Iterators, generators, decorators
- Exception handling

Practical Tasks
- Build a student grade management system
- Implement a Fibonacci sequence generator
- Implement a cached function decorator

### Data Processing & Visualization

- **NumPy (matrix operations)**: matrix operations, broadcasting, random number generation
- **Pandas (data processing)**: DataFrame creation, indexing, grouping, aggregation
- **Matplotlib / Seaborn (data visualization)**: line charts, bar charts, histograms, scatter plots, heatmaps

Practical Tasks
- Analyze Titanic dataset using Pandas to study survival rates
- Plot stock price trends using Matplotlib

### Algorithms & Data Structures

- **Time complexity** & **space complexity**
- Sorting: **Bubble Sort**, **Quick Sort**, **Merge Sort**
- Searching: **Sequential Search**, **Binary Search**, **Hash Table**
- Data structures: **Stack**, **Queue**, **Linked List**, **Heap**
- Graph algorithms: **DFS**, **BFS**, **Shortest Path (Dijkstra, Floyd)**

Practical Tasks
- Implement Quick Sort in Python
- Solve maze shortest path problem using BFS
- Implement shortest path query using Dijkstra’s algorithm

### Computer Fundamentals

- Operating System: **Processes & Threads**, **Memory Management**, **File System**
- Computer Architecture: **CPU vs GPU**, **Parallel Computing**
- Networking basics: **TCP/IP**, **HTTP**

Practical Tasks
- Use `top` in Linux to monitor CPU usage
- Write a Python multithreaded web crawler
- Implement a simple HTTP server

## Stage 3: Machine Learning Fundamentals (6–8 Weeks)

Learning Goals
- Understand and implement basic ML algorithms (Linear Regression, Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, KNN, K-Means)
- Master dimensionality reduction methods (PCA, LDA) and their mathematical principles
- Evaluate models using cross-validation and metrics (accuracy, recall, F1, ROC, AUC)
- Identify overfitting and underfitting, and apply regularization
- Independently perform data preprocessing, feature engineering, and model training
- Design experiments, analyze results, and choose appropriate models

### Core Algorithms

- **Linear Regression**
- **Logistic Regression**
- **Naive Bayes Classifier**
- **Support Vector Machine (SVM)**
- **Decision Tree**, **Random Forest**
- **K-Nearest Neighbors (KNN)**
- Clustering: **K-Means**, **Hierarchical Clustering**
- Dimensionality reduction: **PCA**, **LDA**

### Model Evaluation

- Loss functions: **MSE**, **Cross-Entropy**
- Metrics: accuracy, precision, recall, F1 score, ROC curve, AUC
- Cross-validation, overfitting & regularization (L1/Lasso, L2/Ridge)

Practical Tasks
- Implement Linear Regression & Logistic Regression using Scikit-Learn
- Classify MNIST digits using SVM
- Train and visualize a decision tree on the Titanic dataset

## Stage 4: Deep Learning Core (8–10 Weeks)

Learning Goals
- Master neural network structure (Perceptron, activation functions, loss functions, backpropagation)
- Build and train CNN, RNN, LSTM networks
- Understand and use training techniques (Batch Normalization, Dropout, Learning Rate Scheduling)
- Proficiently use PyTorch/TensorFlow to build models
- Understand mathematical derivation and backpropagation in neural networks
- Train basic tasks like image classification and text classification
- Explain model performance and tune hyperparameters

### Neural Network Basics

- **Perceptron**
- **Multilayer Perceptron (MLP)**
- Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
- Loss functions: MSE, Cross-Entropy
- **Backpropagation**

### Common Networks

- **CNN**
- **RNN**
- **LSTM**
- **GRU**
- **Seq2Seq Models**

### Optimization Techniques

- Batch Normalization
- Dropout
- Learning Rate Scheduler

### Deep Learning Frameworks

- PyTorch / TensorFlow basics
- Model building, training, debugging
- GPU acceleration

Practical Tasks
- Implement a fully connected neural network from scratch using PyTorch
- Train a CNN on CIFAR-10 for classification
- Generate text using RNN

## Stage 5: Advanced LLM (10–12 Weeks)

Learning Goals
- Deeply understand Transformer architecture and self-attention
- Understand pretrained language models (BERT masked LM, GPT causal LM)
- Fine-tune pretrained models using Hugging Face Transformers
- Master LLM optimization & compression techniques (LoRA, knowledge distillation, quantization)

### Traditional NLP

- Text representation: **Bag of Words**, **TF-IDF**
- **Word2Vec**, **GloVe**
- Language models: **n-gram**

### Deep Learning NLP

- RNN/LSTM applications in NLP
- Seq2Seq models and attention mechanism

### Transformer Architecture

- **Self-Attention**
- **Multi-Head Attention**
- **Positional Encoding**
- **Encoder–Decoder structure**
- Pretraining tasks: Masked LM (**BERT**), Causal LM (**GPT**), T5 (**T5**)

### LLM Techniques

- Fine-tuning: **Full Fine-Tuning**, **Low-Rank Adaptation (LoRA)**
- **Parameter-Efficient Fine-Tuning (PEFT)**
- **Knowledge Distillation**
- Model compression & quantization

Practical Tasks
- Implement a Mini-Transformer in PyTorch
- Fine-tune GPT-2 using Hugging Face Transformers for text generation
- Perform text classification with BERT (sentiment analysis)
- Load pretrained models (BERT, GPT-2, LLaMA)
- Implement text generation, Q&A, summarization tasks

## Stage 6: Advanced & Research / Practical (Long-Term)

Learning Goals
- Master cutting-edge LLM techniques (RLHF, RAG, MoE, multimodal)
- Understand distributed training and inference optimization
- Deploy models in real applications (API, web service)
- Conduct research, reproduce papers, and propose improvements

Frontier Research
- **Reinforcement Learning with Human Feedback (RLHF)**
- **Retrieval-Augmented Generation (RAG)**
- **Mixture of Experts (MoE)**
- Multimodal LLMs (vision + text)
- ...

Systems & Engineering
- **Distributed Training** (Data Parallel, Model Parallel, ZeRO)
- Inference acceleration: tensor parallelism, pipeline parallelism, KV caching
- Model deployment: FastAPI, Docker, Kubernetes, TensorRT
- MLOps: version control, monitoring, continuous training

Practical Tasks
- Reproduce experiments from BERT or GPT papers
- Deploy a RAG-based Q&A system
- Fine-tune a large model with LoRA on a small GPU
