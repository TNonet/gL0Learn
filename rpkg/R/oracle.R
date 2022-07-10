#' @title gL0Learn.penalty
#' @description Creates a C++ penalty object. **Warning** This function does not
#' perform value checking and assumes all values conform to `gL0Learn.fit`'s 
#' checks. If this is not true, bare C++ errors or segfaults may occur!
#' The values must satisfy all of the following:
#'    1. `l0`, `l1`, `l2` are of the same type or are 0
#'    2. Type must be real scalar or real matrix
#' @param l0 See `gL0Learn.fit` l0 penalty documentation.
#' @param l1 See `gL0Learn.fit` l0 penalty documentation
#' @param l2 See `gL0Learn.fit` l0 penalty documentation
#' @export
gL0Learn.penalty <- function(l0 = 0, l1 = 0, l2 = 0) {

  is_any_non_zero <- function(x){
    # Is all zero:
    #   (x === 0) OR (x is a matrix and all(x[i, j] === 0))
    # Is any non zero:
    #   (x !== 0) && !(x is a matrix and all(x[i, j] === 0))
      return(!identical(x, 0) && !(is.matrix(x) && all(x == 0)))
  }    
    
  penalty_type <- "double"
  if (is.matrix(l0)) {
    penalty_type <- "mat"
  }
  
  any_non_zero_l1 <- is_any_non_zero(l1)
  any_non_zero_l2 <- is_any_non_zero(l2)
  
  if (any_non_zero_l1 && any_non_zero_l2) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L1L2", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l1, l2))
  } else if (any_non_zero_l1) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L1", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l1))
  } else if (any_non_zero_l2) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L2", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l2))
  } else {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0", penalty_type, sep = "_")))
    return(new(penalty_class, l0))
  }
}

#' @title gL0Learn.bounds
#' @description Creates a C++ bounds object
#'  **Warning** This function does not perform value checking and assumes all 
#'  values conform to `gL0Learn.fit`'s checks. If this is not true, bare C++ 
#'  errors or segfaults may occur!
#'  The values must satisfy all of the following:
#'    1. `lows` and `highs` are of the same type
#'    2. Type must be real scalar or real matrix
#' @param lows See `gL0Learn.fit` l0 penalty documentation.
#' @param highs See `gL0Learn.fit` l0 penalty documentation.
#' @export
gL0Learn.bounds <- function(lows = -Inf, highs = Inf) {
  bound_type <- "double"
  if (is.matrix(lows)) {
    bound_type <- "mat"
  }
  
  is_any_finite <- function(x, sign=-1){
    # Is all infinite:
    #    (x === sign*Inf) OR (x is a matrix AND all(x[i, j] === sign*Inf))
    # Is any finitite:
    #    (x !== sign*Inf) && !(x is a matrix AND all(x[i, j] === sign*Inf))    
    # Claim that if `x` is a matrix, it will most likely not be only Inf
    # Therefore, better to check for any non-Inf than for all Inf.
    # However, we can't demorgans the not in:
    #     !(x is a matrix AND all(x[i, j] === sign*Inf))
    # TODO: Fix this....? Very minor but annoying
      return(!identical(x, sign*Inf) && !(is.matrix(x) && all(x == sign*Inf)))
  } 

  if (is_any_finite(lows) || is_any_finite(highs)) {
    # Claim non-negative bounds will be more popular than non-postiive
    # Therefore, check lows first for non-Inf.
    bounds_class <- eval(parse(text = paste("Bounds", bound_type, sep = "_")))
    return(list(object = new(bounds_class, lows, highs), type = bound_type))
  } else {
    return(list(object = new(NoBounds), type = ""))
  }
}
