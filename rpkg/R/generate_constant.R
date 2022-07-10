#' @import Matrix
#'
#' @title Generate a constantly correlated data set for gL0Learn
#'
#' @description Computes the ...
#' @param n See `gL0Learn.generate_synthetic` for details
#' @param p See `gL0Learn.generate_synthetic` for details
#' @param rho [TODO: Add documentation]
#' @param normalize See `gL0Learn.generate_synthetic` for details
#' @param seed See `gL0Learn.generate_synthetic` for details
#' @export
gL0Learn.generate_constant <- function(n, p, rho, normalize, seed = 1) {
  if ((rho < 0) || (rho > 1)) {
    stop("rho must be in [0, 1]")
  }
  if (p < 2) {
    stop("p must be at least 3")
  }
  set.seed(seed)

  X <- matrix(stats::rnorm(n * p), n, p)

  if (rho != 0) {
    X <- X * sqrt(1 - rho) + sqrt(rho) * matrix(stats::rnorm(n), n, 1)
  }

  sigma <- rho * matrix(1, p, p)
  diag(sigma) <- 1

  pcorr <- -rho / (1 + (p - 2) * rho)
  precision <- 1 / (1 + (p - 1) * rho * pcorr)

  theta <- pcorr * matrix(1, p, p)
  diag(theta) <- 1
  theta <- precision * theta

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
