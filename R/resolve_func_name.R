# import C++ compiled code
#' @useDynLib gL0Learn
#' 
#' @title gL0Learn.resolve_func_name
#'
#' @description Determine proper C++ call based on supplied values
#' @param x The data matrix.

#' @examples
#' MISSING
#' @export
gL0Learn.resolve_func_name <- function(theta,
                                       l0=0,
                                       l1=0,
                                       l2=0,
                                       lows=-Inf,
                                       highs=+Inf,
                                       func='prox'){
    
    params <- vector(mode="list", length=6)
    names(params) <- c("theta", "l0", "l1", "l2", "lows", "highs")
    params[[1]] <- theta
    params[[2]] <- l0
    params[[3]] <- l1
    params[[4]] <- l2
    params[[5]] <- lows
    params[[6]] <- highs
    
    bounds = "Bounds"
    if (identical(lows, -Inf) && identical(highs, +Inf)){
        params[['lows']] <- NULL
        params[['highs']] <- NULL
        lows <- NULL
        highs <- NULL
        bounds = "NoBounds"
    }
    
    penalty = "L0L1L2"
    if (identical(l1, 0)){
        params[['l1']] <- NULL
        l1 <- NULL
        penalty = "L0L2"
    }
    
    for (param_name in names(params)){
        param <- params[[param_name]]
        if (!(gL0Learn.is.real_scalar(param) || gL0Learn.is.real_vector(param))){
            stop(sprintf("expected %s to be a scalar or a vector of scalars but isn't", param_name))
        }
    }
    unique_param_sizes = unique(lapply(params, length))
    unique_param_sizes = unique_param_sizes[unique_param_sizes != 1]
    
    max_param_size = 1
    if (length(unique_param_sizes) > 1L){
        stop("expected parameters to have valid sizes, please see gL0Learn documentation [INSERT LINK]")
    } else if (length(unique_param_sizes) == 1L){
        max_param_size = unique_param_sizes[[1]]
    } # else when length(unique_param_sizes) == 0 is handled by default as max_param_size is set to 1
    
    
    if (length(theta) == 1L){
        if (max_param_size > 1L){
            stop("expected all parametes to be scalars if `theta` is a scalar.")
        }
        theta_type = "double"
    } else {
        theta_type = "vec"
    }
    
    if ((length(l0) == 1L) 
        && (length(l2) == 1L)
        && ((penalty != "L0L1L2") || ((penalty == "L0L1L2") && (length(l1) == 1L)))){
        penalty_type = "Scalar"
    } else if ((length(l0) == max_param_size) 
               && (length(l2) == max_param_size)
               && ((penalty != "L0L1L2") || ((penalty == "L0L1L2") && (length(l1) == max_param_size)))){
        penalty_type = "Vector"
    } else {
        stop("expected all specified penalty values to be either scalars or vectors of the same length")
    }
    
    if (bounds == "NoBounds"){
        bounds_type = ""
    } else if ((length(params[["lows"]]) == 1) && (length(params[["highs"]]) == 1)){
        bounds_type = "Scalar"
    } else if ((length(params[["lows"]]) == max_param_size) && (length(params[["highs"]]) == max_param_size)){
        bounds_type = "Vector"
    } else {
        stop("expected both bounds when specified to be scalars or vectors of the same size, but are not.")
    }

    ## test_Oracle{penalty_type}{penalty}{bounds_type}{bounds}_{func}_{theta_type}
    return(paste("test_Oracle",
                 penalty_type,
                 penalty,
                 bounds_type,
                 bounds,
                 "_", func, "_",
                 theta_type,
                 sep=""))
}



#' @title is.scalar
#'
#' @description Determine if a value is a scalar
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874
#' @param x The value to check for "scalar"-ness

#' @examples
#' MISSING
#' @export
gL0Learn.is.real_scalar <- function(x) {is.atomic(x) && length(x) == 1L && !is.character(x)}

#' @title is.scalar
#'
#' @description Determine if a value is a real matrix with dim `dims`
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874
#' @param x The value to check for "scalar"-ness

#' @examples
#' MISSING
#' @export
gL0Learn.is.real_matrix <- function(x, dims) {is.matrix(x) && is.atomic(x) && identical(dim(x), dims)}

#' @title is.real_vector
#'
#' @description Determine if a value is a vector of scalar values
#' Source: # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r/38088874
#' @param x The value to check for "vector"-ness and element-wise scalar"-ness

#' @examples
#' MISSING
#' @export
gL0Learn.is.real_vector <- function(x) {is.atomic(x) && is.vector(x) && length(x) > 1L  && all(sapply(x, is.real_scalar))}


#' @title is.sympd
#'
#' @description 
#' @param x 

#' @examples
#' MISSING
#' @export
gL0Learn.is.sympd <- function(x) {gL0Learn::is_sympd(x)}

