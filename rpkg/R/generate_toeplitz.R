#' @import Matrix
#'
#' @title Generate a synthetic Toeplitz correlated data set for gL0Learn
#'
#' @param n See `gL0Learn.generate_synthetic` for details
#' @param p See `gL0Learn.generate_synthetic` for details
#' @param rho [TODO: Add documentation]
#' @param normalize See `gL0Learn.generate_synthetic` for details
#' @param seed See `gL0Learn.generate_synthetic` for details
#' @export
gL0Learn.generate_toeplitz <- function(n, p, rho, normalize, seed = 1) {
  if ((rho < 0) || (rho > 1)) {
    stop("rho must be in [0, 1]")
  }
  set.seed(seed)
  X <- matrix(stats::rnorm(n * p), n, p)
  q <- sqrt(1 - rho**2)
  if (rho != 0) {
    for (i in 2:p) {
      X[, i] <- X[, i - 1] * rho + q * X[, i]
    }
  }

  diag_offsets <- abs(replicate(p, 1:p) - t(replicate(p, 1:p)))
  sigma <- rho**diag_offsets
  theta <- matrix(diag_offsets, p, p)
  theta_eq_1 <- theta == 1
  theta_neq_1 <- theta != 1
  theta[theta_eq_1] <- -rho / (1 - rho**2)
  theta[theta_neq_1] <- 0
  diag(theta) <- (1 + rho**2) / (1 - rho**2)
  theta[1, 1] <- 1 / (1 - rho**2)
  theta[p, p] <- 1 / (1 - rho**2)

  if (normalize == "covariance") {
    # Nothing
  } else {
    tiled_theta_diag <- replicate(n, sqrt(diag(theta)))

    X <- X * tiled_theta_diag
    sigma <- sigma * tiled_theta_diag * t(tiled_theta_diag)
    theta <- theta / tiled_theta_diag / t(tiled_theta_diag)
  }
  return(list(X = X, sigma = sigma, theta = theta))
}
