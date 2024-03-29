% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generate_synthetic.R
\name{gL0Learn.generate_synthetic}
\alias{gL0Learn.generate_synthetic}
\title{Generate a synthetic data set for gL0Learn}
\usage{
gL0Learn.generate_synthetic(n, p, model, normalize, seed = 1, ...)
}
\arguments{
\item{n}{The number of observations to generated.
This will create a data matrix of shape (n, p)}

\item{p}{The number of features to generated.
This will create a data matrix of shape (n, p)}

\item{model}{The method for generating datasets.
Currently supported models are:
    1. "KR1": A synthetic Toeplitz correlated data set for gL0Learn
       Must provide additional parameter `rho` through `...`
       See `gL0Learn.generate_synthetic` for details
    2. "independent": An independent correlated data set for gL0Learn
       No additional parameters are needed
    3. "constant": A constantly correlated data set for gL0Learn
       Must provide additional parameter `rho` through `...`
       See `gL0Learn.generate_constant` for details
    4. "banded": A partial banded correlated data set for gL0Learn
       Must provide additional parameter `rho` through `...`
       See `gL0Learn.generate_banded` for details
    5. "regression": A regression data set for gL0Learn
       Must provide additional parameters `rho` and `val` through `...`
       See `gL0Learn.generate_regression` for details}

\item{normalize}{The method for normalizing data
Currently supported normalizaiton methods are:
    1. "covariance": [TODO: Add definition]
    2. "precision": [TODO: Add definition]}

\item{seed}{Seed provided to random number generation for dataset}

\item{...}{Additional parameters needing to be passed to sub 
`gL0Learn.generate_*` functions}
}
\description{
Computes the ...
}
