% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit.R
\name{gL0Learn.gfit}
\alias{gL0Learn.gfit}
\title{Fit an L0-regularized graphical model}
\usage{
gL0Learn.gfit(
  x,
  theta_init = NULL,
  l0 = 0,
  l1 = 0,
  l2 = 0,
  lows = -Inf,
  highs = Inf,
  check_inputs = TRUE,
  max_iter = 100,
  max_active_set_size = 0.1,
  algorithm = "CD",
  tol = 1e-06,
  seed = 1,
  active_set = 0.7,
  super_active_set = 0.5,
  max_swaps = 100,
  shuffle_feature_order = FALSE,
  scale_x = FALSE
)
}
\arguments{
\item{x}{The data matrix of shape (n, p) where each row x[i, ] is believed to
be drawn from N(0, theta)}

\item{theta_init}{The initial guess of theta. Default is the identity matrix.
If provided, must be a symmetric matrix of shape (p, p) such that all
non-zero upper triangle values of `theta_init` are included in `active_set`.
Recommended that `check_inputs` be keep as `True` when providing `theta_init`}

\item{l0}{The L0 regularization penalty.
Must be one of:
    1. Positive scalar. Applies the same L0 regularization to each coordinate
       of `theta`
    2. Symmetric Matrix with only positive values of shape (p, p). Applies
       L0 regularization coordinate by coordinate to `theta`}

\item{l1}{The L1 regularization penalty.
Must be one of:
    1. Positive scalar. Applies the same L1 regularization to each coordinate
       of `theta`
    2. Symmetric Matrix with only positive values of shape (p, p). Applies
       L1 regularization coordinate by coordinate to `theta`}

\item{l2}{The L2 regularization penalty.
Must be one of:
    1. Positive scalar. Applies the same L2 regularization to each coordinate
       of `theta`
    2. Symmetric Matrix with only positive values of shape (p, p). Applies
       L2 regularization coordinate by coordinate to `theta`}

\item{lows}{The minimum value that `theta` can take:
Must be one of:
    1. Non-positive scalar. Applies the same bound to each value of `theta`
    2. Symmetric Matrix with only non-positive values of shape (p, p). 
       Applies bounds coordinate by coordinate to `theta`.
    **Note** Both `highs` and `lows` cannot limit a value of `theta` to 0
    at the same time.}

\item{highs}{The maximum value that `theta` can take:
Must be one of:
    1. Non-negative scalar. Applies the same bound to each value of `theta`
    2. Symmetric Matrix with only non-negative values of shape (p, p). 
       Applies bounds coordinate by coordinate to `theta`.
    **Note** Both `highs` and `lows` cannot limit a value of `theta` to 0
    at the same time.}

\item{check_inputs}{Flag whether or not to check user provided input
If TRUE, checks inputs for validity.
If FALSE, runs on inputs and may error if values are not valid.
Only use this is speed is required and you know what you are doing.}

\item{max_iter}{The maximum number of iterations the algorithm can make
before exiting. May exit before this number of iterations if convergence is 
found.}

\item{max_active_set_size}{The maximum number of non-zero values in `theta` 
expressed in terms of percentage of p**2
The size of provided `active_set` must be less than this number.}

\item{algorithm}{The type of algorithm used to minimize the objective
function. 
Must be one of:
    1. "CD" A variant of cyclic coordinate descent and runs very fast. 
    2. "CDPSI" performs local combinatorial search on top of CD and typically 
    achieves higher quality solutions (at the expense of increased 
    running time).}

\item{tol}{The tolerance for determining convergence. 
Graphical Models have non standard convergence criteria. 
See [TODO: Convergence Documentation] for more details.}

\item{seed}{The seed value used to set randomness. 
The same input values with the same seed run on the same version of 
`gL0learn` will always result in the same value}

\item{active_set}{The set of coordinates that the local optimization
algorithm quickly iterates as potential support values of theta.
Can be one of:
    1. 'full' -> Every value in the (p, p) theta matrix is looped over every
    iteration at first. The active set may decrease in size from there.
    2. a scalar value, t, will be used to find the values of x'x that have
    an absolute value larger than t. The coordinates of these values are the
    initial active_set.
    3. Integer Matrix of shape (m, 2) encoding for the coordinates of the
    active_set. Row k (active_set[k, ]), corresponds to the coordinate in
    theta, theta[active_set[k, 1], active_set[k, 2]], is in the active_set.
    *NOTE* All rows of active_set must encode for valid upper triangle
    coordinates of theta
    (i.e. all(x>0) and all(x<p+1)).
    *NOTE* The rows of active_set must be lexicographically sorted such that
    active_set[k] < active_set[j] -> k < j.}

\item{super_active_set}{The set of coordinates that the global optimization
algorithm can swap in and out of `active_set`. See `active_set` parameter
for valid values. When evaluated, all items in `active_set` must be contained
in `super_active_set`.}

\item{max_swaps}{The maximum number of swaps the "CDPSI" algorithm will 
perform per iteration.}

\item{shuffle_feature_order}{A boolean flag whether or not to shuffle the 
iteration order of `active_set` when optimizing.}

\item{scale_x}{A boolean flag whether x needs to be scaled by 1/sqrt(n).
If scale_x is false (i.e the matrix is already scaled), the solver will not
save a local copy of x and thus reduce memory usage.}
}
\description{
TODO: Fill in description
}
