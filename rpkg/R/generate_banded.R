#' @import Matrix
#'
#' @title Generate a partial banded correlated data set for gL0Learn
#'
#' @description Computes the ...
#' @param n See `gL0Learn.generate_synthetic` for details
#' @param p See `gL0Learn.generate_synthetic` for details
#' @param rho [TODO: Add documentation]
#' @param normalize See `gL0Learn.generate_synthetic` for details
#' @param seed See `gL0Learn.generate_synthetic` for details
#' @export
gL0Learn.generate_banded <- function(n, p, rho, normalize, seed = 1) {
  if ((-0.5 <= rho) || (rho >= 0.5)) {
    stop("rho must be in [-0.5, 00.5]")
  }
  set.seed(seed)
  X <- matrix(stats::rnorm(n * p), n, p)

  theta <- matrix(0, p, p)
  for (i in 1:p - 1) {
    theta[i, i + 1] <- rho
    theta[i, i] <- 1
    theta[i + 1, i] <- rho
  }
  theta[p, p] <- 1

  sigma <- solve(theta)
  sc <- chol(sigma)
  X <- X %*% sc

  if (normalize == "covariance") {
    tiled_sigma_diag <- replicate(n, sqrt(diag(sigma)))

    X <- X * tiled_sigma_diag
    sigma <- sigma * tiled_sigma_diag * t(tiled_sigma_diag)
    theta <- theta / tiled_sigma_diag / t(tiled_sigma_diag)
  } else {
    # Nothing
  }
  return(list(X = X, sigma = sigma, theta = theta))
}
