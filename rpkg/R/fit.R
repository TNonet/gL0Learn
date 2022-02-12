# import C++ compiled code
#' @useDynLib gL0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom methods is
#' @import Matrix
#' 
#' @title Fit an L0-regularized graphical model
#' 
#' @description Computes the ...
#' @param x The data matrix of shape (n, p) where each row x[i, ] is believed to
#' be drawn from N(0, theta)
#' @param theta_init The initial guess of theta 
#' @param scale_x A boolean flag whether x needs to be scaled by 1/sqrt(n).
#' If scale_x is false (i.e the matrix is scaled), the solver will not save a 
#' local copy of x.
#' @param active_set Can be one of:
#'     1. 'full' -> Every value in the (p, p) theta matrix is looped over every 
#'     iteration at first. The active set may decrease in size from there.
#'     2. a scalar value, t will be used to find the elements of xtx that have 
#'     an absolute value larger than t. These values are the initial active_set.
#'     3. Integer Matrix of shape (m, 2) encoding for the coordinates of the 
#'     active_set. Row k (active_set[k, ]), corresponds to the coordinate in 
#'     theta, theta[active_set[k, 1], active_set[k, 2]], is in the active_set.
#'     *NOTE* All rows of active_set must encode for valid coordinates of theta
#'     (i.e. all(x>0) and all(x<p+1)).
#'     *NOTE* The rows of active_set must be lexicographically sorted such that
#'     active_set[k] < active_set[j] -> k < j.
#'     
#' @param super_active_set Can be any value that active_set can be.
#' @param check_inputs Not implemented atm. TODO:
#'     If TRUE, checks inputs for validity. If not, runs on inputs and may error.
#'     Only use this is speed is required and you know what you are doing.
#' @export
gL0Learn.gfit <- function(x,
                          theta_init=NULL,
                          l0=0,
                          l1=0,
                          l2=0,
                          lows=-Inf,
                          highs=Inf,
                          max_iter=100,
                          algorithm="CD",
                          atol=1e-6,
                          rtol=1e-6,
                          initial_active_set=0.7,
                          super_active_set=0.5,
                          swap_iters=NULL,
                          scale_x=FALSE){
    
    x_dims <- dim(x)
    if (length(x_dims) != 2){
        stop("L0Learn.gfit requires x to be a 2D array type")
    }
    n <- x_dims[[1]]
    p <- x_dims[[2]]
    
    if (p < 1){
        stop("L0Learn.gfit requires x to have atleast 2 columns")
    }
    
    y <- NULL
    if (scale_x){
        y <- x/sqrt(n)
    } else {
        y <- x
    }
    
    if (is.null(theta_init)){
        theta_init <- diag(p)
    } else if (dim(theta_init) != c(p, p) || ! gL0Learn.is.sympd(theta_init)){
        stop("expected theta_init to be NULL or a semi-positive-definite matrix of side length p")
    }
    
    if (gL0Learn.is.real_scalar(l0) && gL0Learn.is.real_scalar(l2)){
        if (!gL0Learn.is.real_scalar(l1)){
          stop("expected that l1 be a scalar if l0 and l2 are.")
        } else if (l1 == 0){
          l1 <- NULL
        }
    } else if (gL0Learn.is.real_matrix(l0, c(p, p)) && gL0Learn.is.real_matrix(l2, c(p, p))){
        if (!(gL0Learn.is.real_scalar(l1) || gL0Learn.is.real_matrix(l0, c(p, p)))){
          stop("expected that l1 be a matrix of dims (p, p) if l0 and l2 are.")
        } else if (l1 == 0){
          l1 <- NULL
        }
    }
    
    if (gL0Learn.is.real_scalar(lows) && gL0Learn.is.real_scalar(highs)){
      if (lows != -Inf || highs != Inf){
        lows <- NULL
        highs <- NULL
      } else if (lows > 0 || highs < 0){
        stop("exected lows to be less than or equal to zero and highs to be greater than or equal to 0")
      }
    } else if (!(gL0Learn.is.real_matrix(lows, c(p, p)) && gL0Learn.is.real_matrix(highs, c(p, p)))){
      stop("expected that lows or highs be scalar values or matricies of dims (p, p).")
    }
      
    if (max_iter < 1){
      stop("expected max_iter to be a positive integer, but isn't")
    }
    
    if (!(algorithm %in% c("CD", "CDPSI"))){
      stop("expected algorithm to be a `CD` or `CDPSI`, but isn't")
    }
      
    
    if (atol < 0){
      stop("expected atol to be a positive number, but isn't")
    }
    
    if (rtol < 0 || rtol >= 1){
      stop("expected rtol to be a number between 0 and 1 (exlusive), but isn't.")
    }
    
    
    # Notes:
    #   1. initial_active_set and super_active_set will enter as (K, 2) integer matricies
    #   2. initial_active_set must be a subset of super_active_set
    #   3. support of non-zero non-diagonal values of theta_init must be a subset of initial_active_set
    #   
    
    initial_active_set <- check_make_valid_coordinate_matrix('initial_active_set', initial_active_set, y)
    super_active_set <- check_make_valid_coordinate_matrix('super_active_set', super_active_set, y)
    if (!check_is_valid_coordinate_subset(super_active_set, initial_active_set)){
      stop("expected `initial_active_set` be a subset of `super_active_set`, 
           but is not. Please see documentation on how `super_active_set` and
           `initial_active_set` are selected")
    }
    
    theta_init_support <- test_unravel_indices(which(t(theta_init*upper.tri(theta_init)) != 0, arr.ind=FALSE) - 1, p)
    
    if (!check_is_valid_coordinate_subset(initial_active_set, theta_init_support)){
      stop("expected support of `theta_init` to be a subset of `initial_active_set`, 
           but is not. Please see documentation on how `theta_init` and
           `initial_active_set` are selected")
    }
    
    return(gL0Learn_fit_R(y, theta_init, l0, l1, l2, algorithm, lows, highs, initial_active_set, super_active_set, atol, rtol, max_iter))
  
}

check_make_valid_coordinate_matrix <- function(parameter_name, coordinate_matrix, y){
  if (gL0Learn.is.real_scalar(coordinate_matrix) && coordinate_matrix >= 0){
    return(test_union_of_correlated_features2(y, coordinate_matrix))
  } else if (coordinate_matrix == 'full') {
    return(upper_triangluar_coords(p))
  } else if (length(dim(coordinate_matrix)) == 2){
    set_dim = dim(coordinate_matrix)
    set_length <- set_dim[[1]]
    should_be_two <- set_length[[2]]
    if (should_be_two != 2){
      stop(sprintf("expected `%s` to be a N by 2 integer matrix, 
                   but is N by %s", 
                   parameter_name, as.character(should_be_two)))
    }
    if (!check_coordinate_matrix_is_valid(coordinate_matrix)){
      stop(sprintf("expected `%s` to be sorted lexographically and refer to
                   only upper triangle values, but is not.", parameter_name))
    }
    return(coordinate_matrix)
  }

  stop(sprintf("Unable to determine `%s` meaning. See documentation for
         possible values of `%s`", parameter_name, parameter_name))

}
    