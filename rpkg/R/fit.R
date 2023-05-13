# import C++ compiled code
#' @useDynLib gL0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom methods is
#' @import Matrix
#' @title Fit an L0-regularized graphical model
#' @description TODO: Fill in description
#' @param x The data matrix of shape (n, p) where each row x[i, ] is believed to
#' be drawn from N(0, theta)
#' @param theta_init The initial guess of theta. Default is the identity matrix.
#' If provided, must be a symmetric matrix of shape (p, p) such that all
#' non-zero upper triangle values of `theta_init` are included in `active_set`.
#' Recommended that `check_inputs` be keep as `True` when providing `theta_init`
#' @param l0 The L0 regularization penalty.
#' Must be one of:
#'     1. Positive scalar. Applies the same L0 regularization to each coordinate
#'        of `theta`
#'     2. Symmetric Matrix with only positive values of shape (p, p). Applies
#'        L0 regularization coordinate by coordinate to `theta`
#' @param l1 The L1 regularization penalty.
#' Must be one of:
#'     1. Positive scalar. Applies the same L1 regularization to each coordinate
#'        of `theta`
#'     2. Symmetric Matrix with only positive values of shape (p, p). Applies
#'        L1 regularization coordinate by coordinate to `theta`
#' @param l2 The L2 regularization penalty.
#' Must be one of:
#'     1. Positive scalar. Applies the same L2 regularization to each coordinate
#'        of `theta`
#'     2. Symmetric Matrix with only positive values of shape (p, p). Applies
#'        L2 regularization coordinate by coordinate to `theta`
#' @param highs The maximum value that `theta` can take:
#' Must be one of:
#'     1. Non-negative scalar. Applies the same bound to each value of `theta`
#'        This will ensure that every value of theta will respect:
#'            theta[i , j] <= highs for i, j in 0 to p-1
#'     2. Symmetric Matrix with only non-negative values of shape (p, p).
#'        Applies bounds coordinate by coordinate to `theta`.
#'        This will ensure that every value of theta will respect:
#'            theta[i , j] <= highs[i, j] for i, j in 0 to p-1
#'     **Note** Both `highs` and `lows` cannot limit a value of `theta` to 0
#'     at the same time.
#' @param lows The minimum value that `theta` can take:
#' Must be one of:
#'     1. Non-positive scalar. Applies the same bound to each value of `theta`
#'        This will ensure that every value of theta will respect:
#'            theta[i , j] >= lows for i, j in 0 to p-1
#'     2. Symmetric Matrix with only non-positive values of shape (p, p).
#'        Applies bounds coordinate by coordinate to `theta`.
#'        This will ensure that every value of theta will respect:
#'            theta[i , j] >= lows[i, j] for i, j in 0 to p-1
#'     **Note** Both `highs` and `lows` cannot limit a value of `theta` to 0
#'     at the same time.
#' @param check_inputs Flag whether or not to check user provided input
#'     If TRUE, checks inputs for validity.
#'     If FALSE, runs on inputs and may error if values are not valid.
#'     Only use this is speed is required and you know what you are doing.
#' @param max_iter The maximum number of iterations the algorithm can make
#' before exiting. May exit before this number of iterations if convergence is
#' found.
#' @param max_active_set_size The maximum number of non-zero values in `theta`
#' expressed in terms of percentage of p**2
#' The size of provided `active_set` must be less than this number.
#' @param algorithm The type of algorithm used to minimize the objective
#' function.
#' Must be one of:
#'     1. "CD" A variant of cyclic coordinate descent and runs very fast.
#'     2. "CDPSI" performs local combinatorial search on top of CD and typically
#'     achieves higher quality solutions (at the expense of increased
#'     running time).
#' @param tol The tolerance for determining convergence.
#' Graphical Models have non standard convergence criteria.
#' See [TODO: Convergence Documentation] for more details.
#' @param seed The seed value used to set randomness.
#' The same input values with the same seed run on the same version of
#' `gL0learn` will always result in the same value
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
#' algorithm can swap in and out of `active_set`. See `active_set` parameter
#' for valid values. When evaluated, all items in `active_set` must be contained
#' in `super_active_set`.
#' @param max_swaps The maximum number of swaps the "CDPSI" algorithm will
#' perform per iteration.
#' @param shuffle_feature_order A boolean flag whether or not to shuffle the
#' iteration order of `active_set` when optimizing.
#' @param scale_x A boolean flag whether x needs to be scaled by 1/sqrt(n).
#' If scale_x is false (i.e the matrix is already scaled), the solver will not
#' save a local copy of x and thus reduce memory usage.
#' @export
gL0Learn.gfit <- function(x, # nolint
                          theta_init = NULL,
                          l0 = 0,
                          l1 = 0,
                          l2 = 0,
                          lows = -Inf,
                          highs = Inf,
                          check_inputs = TRUE,
                          max_iter = 100,
                          max_active_set_size = .1,
                          algorithm = "CD",
                          tol = 1e-6,
                          seed = 1,
                          active_set = 0.7,
                          super_active_set = 0.5,
                          max_swaps = 100,
                          shuffle_feature_order = FALSE,
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
  } else if (check_inputs && (dim(theta_init) != c(p, p) || !all(x == t(x)))) {
    stop("expected theta_init to be NULL or symmetric matrix of side length p")
  }

  if (check_inputs) {
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
  }


  if (max_iter < 1) {
    stop("expected max_iter to be a positive integer, but isn't")
  }

  if (!(algorithm %in% c("CD", "CDPSI"))) {
    stop("expected algorithm to be a `CD` or `CDPSI`, but isn't")
  }

  if (tol < 0) {
    stop("expected atol to be a positive number, but isn't")
  }

  penalty <- gL0Learn.penalty(l0, l1, l2)
  if (check_inputs && !penalty$validate()) {
    stop("Penalty values are invalid see DOCUEMENTATION")
  }

  if (check_inputs && !(
    (gL0Learn.is.real_scalar(lows) && gL0Learn.is.real_scalar(highs)) ||
      (gL0Learn.is.real_matrix(lows, c(p, p)) && gL0Learn.is.real_matrix(highs, c(p, p)))
  )
  ) {
    stop("expected that lows and highs be either scalar values or matricies of dims (p, p).")
  }

  bounds <- gL0Learn.bounds(lows, highs)
  if (check_inputs && !bounds$object$validate()) {
    stop("Bounds are invalid. SEE DOCUMENTATION ")
  }

  # Notes:
  #   1. active_set and super_active_set will enter as
  #     (K, 2) integer matrices
  #   2. active_set must be a subset of super_active_set
  #   3. support of non-zero non-diagonal values of theta_init must be a
  #     subset of active_set

  active_set <- check_make_valid_coordinate_matrix(
    active_set, y, "active_set",
    check_inputs = check_inputs
  )

  super_active_set <- check_make_valid_coordinate_matrix(
    super_active_set, y, "super_active_set",
    check_inputs = check_inputs
  )

  if (check_inputs && !check_is_valid_coordinate_subset(
    super_active_set,
    active_set
  )) {
    stop("expected `active_set` be a subset of `super_active_set`,
           but is not. Please see documentation on how `super_active_set` and
           `active_set` are selected")
  }

  if (check_inputs) {
    theta_init_support <- test_unravel_indices(
      which(t(theta_init * upper.tri(theta_init)) != 0, arr.ind = FALSE) - 1, p
    )

    if (!check_is_valid_coordinate_subset(
      active_set,
      theta_init_support
    )) {
      stop("expected support of `theta_init` to be a subset of
      `active_set`, but is not. Please see documentation on how
      `theta_init` and `active_set` are selected")
    }
  }

  fit_bound_method <- eval(parse(text = paste("penalty$fit", bounds$type, sep = "_")))

  return(fit_bound_method(
    y,
    theta_init,
    algorithm,
    bounds$object,
    active_set,
    super_active_set,
    tol,
    max_active_set_size,
    max_iter,
    seed,
    max_swaps,
    shuffle_feature_order
  ))
}

# import C++ compiled code
#' @useDynLib gL0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom methods is
#' @import Matrix
#' @title Make or Check coordinate matrix
#' @description TODO: Fill in description
#' @param coordinate_matrix
#' Must be one of:
#'     1. Non-negative scalar: Indicates that the coordinate matrix should be
#'     coordinates (i, j) of yty st |yty[i, j]| >= t
#'     2. "full": Indicates that the coordinate matrix should be the full
#'     upper triangle coordinate matrix
#'     3. Integer matrix of shape (m, 2) such that coordinate_matrix[i, ] is
#'     a coordinate (i, j). Must be sorted lexigraphically.
#' @param y The scaled data matrix of shape (n, p) where each row x[i, ],
#'  x being the un-scaled data matrix, is believed to be drawn from N(0, theta)
#'  Only used when `coordinate_matrix` is a non-negative scalar.
#' @param parameter_name The name of the parameter being checked.
#' Used for error propagation
#' @param check_inputs Flag whether or not to check user provided input
#'     If TRUE, checks `coordinate_matrix` for validity.
#'     If FALSE, does not check `coordinate_matrix` for validity.
#' @export
check_make_valid_coordinate_matrix <- function(coordinate_matrix,
                                               y,
                                               parameter_name = "",
                                               check_inputs = TRUE) {
  y_dims <- dim(y)
  p <- y_dims[[2]]
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
    if (check_inputs && !check_coordinate_matrix_is_valid(coordinate_matrix)) {
      stop(sprintf("expected `%s` to be sorted lexographically and refer to
                   only upper triangle values, but is not.", parameter_name))
    }
    return(coordinate_matrix)
  }

  stop(sprintf("Unable to determine `%s` meaning. See documentation for
         possible values of `%s`", parameter_name, parameter_name))
}
