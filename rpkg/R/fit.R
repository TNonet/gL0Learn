# import C++ compiled code
#' @useDynLib gL0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom methods is
#' @import Matrix
#' @title Fit an L0-regularized graphical model
#' @description Computes the ...
#' @param x The data matrix of shape (n, p) where each row x[i, ] is believed to
#' be drawn from N(0, theta)
#' @param theta_init The initial guess of theta.
#' Defaults to the identity matrix.
#' If provided, must be a symmetric matrix of shape (p, p) such that all
#' non-zero upper triangle values of `theta_init` are included in `active_set`.
#' @param scale_x A boolean flag whether x needs to be scaled by 1/sqrt(n).
#' If scale_x is false (i.e the matrix is already scaled), the solver will not
#' save a local copy of x and thus reduce memory usage.
#' @param active_set The set of coordinates that the local optimization
#' algorithm quickly iterates as potential support values of theta.
#' Can be one of:
#'     1. 'full' -> Every value in the (p, p) theta matrix is looped over every
#'     iteration at first. The active set may decrease in size from there.
#'     2. a scalar value, t, will be used to find the values of x'x that have
#'     an absolute value larger than t. The coordinates of these values are the
#'     initial active_set.
#'     3. Integer Matrix of shape (m, 2) encoding for the coordinates of the
#'     active_set. Row k (active_set[k, ]), corresponds to the coordinate in
#'     theta, theta[active_set[k, 1], active_set[k, 2]], is in the active_set.
#'     *NOTE* All rows of active_set must encode for valid upper triangle
#'     coordinates of theta
#'     (i.e. all(x>0) and all(x<p+1)).
#'     *NOTE* The rows of active_set must be lexicographically sorted such that
#'     active_set[k] < active_set[j] -> k < j.
#' @param super_active_set The set of coordinates that the global optimization
#' algorithim can swap in and out of `active_set`.
#' Can be any value that is provided to `active_set`.
#' Must be larger or equal to `active_set`.
#' @param check_inputs Not implemented atm. TODO:
#'     If TRUE, checks inputs for validity.
#'     If FALSE, runs on inputs and may error if values are not valid.
#'     Only use this is speed is required and you know what you are doing.
#' @export
gL0Learn.gfit <- function(x, # nolint
                          theta_init = NULL,
                          l0 = 0,
                          l1 = 0,
                          l2 = 0,
                          lows = -Inf,
                          highs = Inf,
                          max_iter = 100,
                          max_active_set_size = .1,
                          algorithm = "CD",
                          atol = 1e-6,
                          rtol = 1e-6,
                          initial_active_set = 0.7,
                          super_active_set = 0.5,
                          swap_iters = NULL,
                          scale_x = FALSE) {
  x_dims <- dim(x)
  if (length(x_dims) != 2) {
    stop("L0Learn.gfit requires x to be a 2D array type")
  }
  n <- x_dims[[1]]
  p <- x_dims[[2]]

  if (p < 1) {
    stop("L0Learn.gfit requires x to have atleast 2 columns")
  }

  y <- NULL
  if (scale_x) {
    y <- x / sqrt(n)
  } else {
    y <- x
  }

  if (max_active_set_size <= 1) {
    # TODO, Error check on `max_active_set_size`
    max_active_set_size <- as.integer(max_active_set_size * p**2)
  }

  if (is.null(theta_init)) {
    theta_init <- diag(p)
  } else if (dim(theta_init) != c(p, p) || !gL0Learn.is.sympd(theta_init)) {
    stop("expected theta_init to be NULL or a semi-positive-definite
             matrix of side length p")
  }

  if (gL0Learn.is.real_scalar(l0)) {
    if (!gL0Learn.is.real_scalar(l1)) {
      stop("expected that l1 be a scalar if l0 is.")
    }
    if (!gL0Learn.is.real_scalar(l2)) {
      stop("expected that l2 be a scalar if l0 is.")
    }
  } else if (gL0Learn.is.real_matrix(l0, c(p, p))) {
    if (!(gL0Learn.is.real_matrix(l1, c(p, p)) || (gL0Learn.is.real_scalar(l1) && l1 == 0))) {
      stop("expected that l1 be a matrix of dims (p, p) or scalar 0 if l0 is a matrix.")
      if (gL0Learn.is.real_scalar(l1)) {
        l1 <- matrix(rep(0, p * p), p, p)
      }
    }
    if (!(gL0Learn.is.real_matrix(l2, c(p, p)) || (gL0Learn.is.real_scalar(l2) && l2 == 0))) {
      stop("expected that l2 be a matrix of dims (p, p) or scalar 0 if l0 is a matrix.")
      if (gL0Learn.is.real_scalar(l2)) {
        l2 <- matrix(rep(0, p * p), p, p)
      }
    }
  } else {
    # TODO: Improve wording here.
    stop("expected that l0, l1, and l2 statisfy one of the following two conditions:
           1) All non-negative scalars
           2) l0 is a non-negative matricies of dims (p, p)
           and l1 and l2 are either 0 or non-negaitve matricies of dims (p, p)")
  }

  if (max_iter < 1) {
    stop("expected max_iter to be a positive integer, but isn't")
  }

  if (!(algorithm %in% c("CD", "CDPSI"))) {
    stop("expected algorithm to be a `CD` or `CDPSI`, but isn't")
  }

  if (atol < 0) {
    stop("expected atol to be a positive number, but isn't")
  }

  if (rtol < 0 || rtol >= 1) {
    stop("expected rtol to be a number between 0 and 1 (exlusive),
           but isn't.")
  }

  penalty <- gL0Learn.penalty(l0, l1, l2)
  if (!penalty$validate()) {
    stop("Penalty values are invalid see DOCUEMENTATION")
  }

  if (!(
    (gL0Learn.is.real_scalar(lows) && gL0Learn.is.real_scalar(highs)) ||
      (gL0Learn.is.real_matrix(lows, c(p, p)) && gL0Learn.is.real_matrix(highs, c(p, p)))
  )
  ) {
    stop("expected that lows and highs be either scalar values or matricies of dims (p, p).")
  }

  bounds <- gL0Learn.bounds(lows, highs)
  if (!bounds$object$validate()) {
    stop("Bounds are invalid. SEE DOCUMENTATION ")
  }

  # Notes:
  #   1. initial_active_set and super_active_set will enter as
  #     (K, 2) integer matrices
  #   2. initial_active_set must be a subset of super_active_set
  #   3. support of non-zero non-diagonal values of theta_init must be a
  #     subset of initial_active_set

  initial_active_set <- check_make_valid_coordinate_matrix(
    "initial_active_set", initial_active_set, y, p
  )

  super_active_set <- check_make_valid_coordinate_matrix(
    "super_active_set", super_active_set, y, p
  )

  if (!check_is_valid_coordinate_subset(
    super_active_set,
    initial_active_set
  )) {
    stop("expected `initial_active_set` be a subset of `super_active_set`,
           but is not. Please see documentation on how `super_active_set` and
           `initial_active_set` are selected")
  }

  theta_init_support <- test_unravel_indices(
    which(t(theta_init * upper.tri(theta_init)) != 0, arr.ind = FALSE) - 1, p
  )

  if (!check_is_valid_coordinate_subset(
    initial_active_set,
    theta_init_support
  )) {
    stop("expected support of `theta_init` to be a subset of
      `initial_active_set`, but is not. Please see documentation on how
      `theta_init` and `initial_active_set` are selected")
  }

  tol <- 1e-6
  seed <- 1
  max_swaps <- 100
  shuffle_feature_order <- FALSE

  fit_bound_method <- eval(parse(text = paste("penalty$fit", bounds$type, sep = "_")))

  return(fit_bound_method(
    y,
    theta_init,
    algorithm,
    bounds$object,
    initial_active_set,
    super_active_set,
    tol,
    max_active_set_size,
    max_iter,
    seed,
    max_swaps,
    shuffle_feature_order
  ))
}



check_make_valid_coordinate_matrix <- function(parameter_name,
                                               coordinate_matrix,
                                               y,
                                               p) {
  if (gL0Learn.is.real_scalar(coordinate_matrix) && coordinate_matrix >= 0) {
    return(test_union_of_correlated_features2(y, coordinate_matrix))
  } else if (coordinate_matrix == "full") {
    return(upper_triangluar_coords(p))
  } else if (length(dim(coordinate_matrix)) == 2) {
    set_dim <- dim(coordinate_matrix)
    set_length <- set_dim[[1]]
    should_be_two <- set_length[[2]]
    if (should_be_two != 2) {
      stop(sprintf(
        "expected `%s` to be a N by 2 integer matrix,
                   but is N by %s",
        parameter_name, as.character(should_be_two)
      ))
    }
    if (!check_coordinate_matrix_is_valid(coordinate_matrix)) {
      stop(sprintf("expected `%s` to be sorted lexographically and refer to
                   only upper triangle values, but is not.", parameter_name))
    }
    return(coordinate_matrix)
  }

  stop(sprintf("Unable to determine `%s` meaning. See documentation for
         possible values of `%s`", parameter_name, parameter_name))
}
