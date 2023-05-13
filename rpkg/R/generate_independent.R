#' @import Matrix
#'
#' @title Generate a independent correlated data set for gL0Learn
#'
#' @description Computes the the theta matrix, sigma matrix, and sampled X
#' matrix for an independent correlated graphical data set.
#' @param n See `gL0Learn.generate_synthetic` for details
#' @param p See `gL0Learn.generate_synthetic` for details
#' @param normalize See `gL0Learn.generate_synthetic` for details
#' @param seed See `gL0Learn.generate_synthetic` for details
#' @export
gL0Learn.generate_independent <- function(n, p, normalize, seed = 1) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * p), n, p)
  sigma <- diag(p)
  theta <- diag(p)
  return(list(X = X, sigma = sigma, theta = theta))
}
