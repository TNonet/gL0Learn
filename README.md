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

The toolkit is implemented in C++14 using armadillo as the linear algebra package. We have provided easy-to-use R and Python interfaces that provide efficient wrappers around the C++ library; See the section below for summary of the installation.


## R Package Installation and Summary
More detailed documentaiton on the _gL0Learn_ R package can be found here []. The latest version can be installed from CRAN as follows:
```R
install.packages("gL0Learn", repos="https://cran.rstudio.com")
```
Alternatively, _gL0Learn_ can also be installed from Github as follows. Building `gL0Learn` from source may require the installation of libraries that R cannot install. More details on installing `gL0Learn` from source can be found here []
```R
library(devtools)
install_github("tnonet/gL0Learn", subdir="rpkg")
```

## Python Pacakge Installation and Summary
More detailed documentaiton on the _gL0Learn_ Python package can be found here []. The latest version can be installed from PyPi as follows:
```bash
pip install gL0Learn
```
Alternatively, _gL0Learn_ can also be installed from Github as follows. Building `gL0Learn` from source may require the installation of libraries that Python/pip cannot install. More details on installing `gL0Learn` from source can be found here []

### Linux/MacOS
```bash
python -m pip install "gl0Learn @ git+https://github.com/TNonet/gL0Learn#subdirectory=pypkg"
```
### Windows
```bash
py -m pip install "gl0Learn @ git+https://github.com/TNonet/gL0Learn#subdirectory=pypkg"
```
