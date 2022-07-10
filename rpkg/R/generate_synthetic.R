#' @import Matrix
#'
#' @title Generate a synthetic data set for gL0Learn
#'
#' @description Computes the ...
#' @param n The number of observations to generated.
#' This will create a data matrix of shape (n, p)
#' @param p The number of features to generated.
#' This will create a data matrix of shape (n, p)
#' @param model The method for generating datasets.
#' Currently supported models are:
#'     1. "KR1": A synthetic Toeplitz correlated data set for gL0Learn
#'        Must provide additional parameter `rho` through `...`
#'        See `gL0Learn.generate_synthetic` for details
#'     2. "independent": An independent correlated data set for gL0Learn
#'        No additional parameters are needed
#'     3. "constant": A constantly correlated data set for gL0Learn
#'        Must provide additional parameter `rho` through `...`
#'        See `gL0Learn.generate_constant` for details
#'     4. "banded": A partial banded correlated data set for gL0Learn
#'        Must provide additional parameter `rho` through `...`
#'        See `gL0Learn.generate_banded` for details
#'     5. "regression": A regression data set for gL0Learn
#'        Must provide additional parameters `rho` and `val` through `...`
#'        See `gL0Learn.generate_regression` for details
#' @param normalize The method for normalizing data
#' Currently supported normalizaiton methods are:
#'     1. "covariance": [TODO: Add definition]
#'     2. "precision": [TODO: Add definition]
#' @param seed Seed provided to random number generation for dataset
#' @param ... Additional parameters needing to be passed to sub 
#' `gL0Learn.generate_*` functions
#' @export
gL0Learn.generate_synthetic <- function(n, p, model, normalize, seed = 1, ...) {
  if (!(normalize %in% list("precision", "covariance"))) {
    stop("normalize must be either 'precision' or  'covariance'")
  }

  if (model == "KR1") {
    args <- as.list(match.call())
    if (is.null(args$rho)) {
      stop("When model is 'KR1',
                 an additional parameter 'rho' is needed.")
    }
    return(gL0Learn.generate_toeplitz(n,
      p,
      rho = args$rho,
      normalize = normalize,
      seed = seed
    ))
  } else if (model == "independent") {
    return(gL0Learn.generate_independent(n,
      p,
      normalize = normalize,
      seed = seed
    ))
  } else if (model == "constant") {
    args <- as.list(match.call())
    if (is.null(args$rho)) {
      stop("When model is 'constant',
                 an additional parameter 'rho' is needed.")
    }
    return(gL0Learn.generate_constant(n,
      p,
      rho = args$rho,
      normalize = normalize,
      seed = seed
    ))
  } else if (model == "banded") {
    args <- as.list(match.call())
    if (is.null(args$rho)) {
      stop("When model is 'banded',
                 an additional parameter 'rho' is needed.")
    }
    return(gL0Learn.generate_banded(n,
      p,
      rho = args$rho,
      normalize = normalize,
      seed = seed
    ))
  } else if (model == "regression") {
    args <- as.list(match.call())
    if (is.null(args$k)) {
      stop("When model is 'banded',
                 an additional parameter 'k' is needed.")
    }
    if (is.null(args$val)) {
      stop("When model is 'banded',
                 an additional parameter 'val' is needed.")
    }
    return(gL0Learn.generate_banded(n,
      p,
      rho = args$rho,
      normalize = normalize,
      seed = seed
    ))
  } else {
    stop("model is not supported.")
  }
}
