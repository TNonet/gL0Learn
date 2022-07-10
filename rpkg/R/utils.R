#' @title gL0Learn.is.scalar
#'
#' @description Determine if a value is a scalar
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874 # nolint
#' @param x The value to check for "scalar"-ness
#' @export
gL0Learn.is.real_scalar <- function(x) {
  return(is.atomic(x) && length(x) == 1L && !is.character(x))
}

#' @title gL0Learn.is.real_matrix
#'
#' @description Determine if a value is a real matrix with dim `dims`
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874 # nolint
#' @param x The matrix to check for realness (i.e x should exist in |R^{m, n})
#' @param dims the dimension of the matrix (m, n)

#' @export
gL0Learn.is.real_matrix <- function(x, dims) {
  return(is.matrix(x) && is.atomic(x) && all(dim(x) == dims))
}

#' @title gL0Learn.is.real_vector
#'
#' @description Determine if a value is a vector of scalar values
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874 # nolint
#' @param x The value to check for "vector"-ness and element-wise scalar"-ness
#' 
#' @export
gL0Learn.is.real_vector <- function(x) {
  return(is.atomic(x) &&
    is.vector(x) &&
    length(x) > 1L &&
    all(sapply(x, gL0Learn.is.real_scalar)))
}
