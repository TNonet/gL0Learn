#' @title gL0Learn.penalty
#' @description Creates a C++ oracle object
#' @param l0
#' @param l1
#' @param l2
#' @examples
#' MISSING
#' @export
NULL
gL0Learn.penalty <- function(l0 = 0, l1 = 0, l2 = 0) {
  penalty_type <- "double"
  if (is.matrix(l0)) {
    penalty_type <- "mat"
  }
  if (any(l1) && any(l2)) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L1L2", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l1, l2))
  } else if (any(l1)) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L1", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l1))
  } else if (any(l2)) {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0L2", penalty_type, sep = "_")))
    return(new(penalty_class, l0, l2))
  } else {
    penalty_class <- eval(parse(text = paste("WrappedPenaltyL0", penalty_type, sep = "_")))
    return(new(penalty_class, l0))
  }
}

#' @title gL0Learn.bounds
#' @description Creates a C++ bounds object
#' Assume inputs have already been sanitized to satisfied the following constraints:
#' 1. `lows` and `highs` are of the same type
#' 2. Type must be real scalar or real matrix
#' @param lows
#' @param highs
#' @examples
#' MISSING
#' @export
NULL
gL0Learn.bounds <- function(lows = -Inf, highs = Inf) {
  bound_type <- "double"
  if (is.matrix(lows)) {
    bound_type <- "mat"
  }

  if (any(is.finite(lows)) && any(is.finite(highs))) {
    bounds_class <- eval(parse(text = paste("Bounds", bound_type, sep = "_")))
    return(list(object = new(bounds_class, lows, highs), type = bound_type))
  } else {
    return(list(object = new(NoBounds), type = ""))
  }
}
