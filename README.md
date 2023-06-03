# gL0Learn: Graphical Fast Best Subset Selection
Efficient Algorithms for L0 Regularized Graphical Learning

### Wenyu Chen, Kayhan Behdin, Rahul Mazumder, and Tim Nonet
### Massachusetts Institute of Technology

## Introduction
_gL0Learn_ is a highly efficient framework for solving L0-regularized graphical learning probelsm. It can (approximately) solve the following problem where the empirical loss is penalized by combinations of the L0, L1, and L2 norms:

Given the data matrix $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ with each row $\boldsymbol{x}^{(i)} \sim N(0,\boldsymbol{\Theta^{-1}})$. 
Let $\boldsymbol{Y} = \frac{1}{\sqrt{n}}\boldsymbol{X}$ and $\boldsymbol{S} \in \mathbb{R}^{p \times p }$ such that $\boldsymbol{S} = \frac{1}{n}\boldsymbol{X}^T\boldsymbol{X} = \boldsymbol{Y}^T\boldsymbol{Y}$ be the sample covariance matrix.

Let $\theta_i \in \mathbb{R}^p$ be $i$-th column of $\boldsymbol{\Theta}$, and $\theta_{ij}$ be the $(i,j)$-th element of $\boldsymbol{\Theta}$. Let $\boldsymbol{y}_i$ be $i$-th column of $\boldsymbol{X}$.

![Alt text](/docs/images/symmetric-pseudo-likelihood-loss-function-with-reguralization.png)
