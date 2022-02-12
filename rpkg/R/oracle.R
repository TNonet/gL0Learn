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
#' 
#' @export
gL0Learn.oracle_prox <- function(theta,
                                 l0=0,
                                 l1=0,
                                 l2=0,
                                 lows=-Inf,
                                 highs=Inf){
    
    # Three options, scalar, vector, matrix
    # Attempt to return the same type that was sent it
    # Therefore, we must identify when the item is each by the following rules:
    # scalar <- NULL dim, 1 length
    # vector <- NULL dim, >1 length
    # matrix <- None NULL dim, any length
    # fail otherwise.
    
    data_type <- function(v){
        dim_type <- dim(v)
        len <- length(v)
        
        if (is.null(dim_type)) {
            if (len == 1L){
                return("SCALAR")
            } else if (len > 1L) {
                return("VECTOR")
            } else {
                stop("Expected scalar, vector, or matrix but got something else.")
            }
        } else {
            return("MATRIX")
        }
    }
    
    is_same_type <- function(v1, v2){
        return(data_type(v1) == data_type(v2))
    }
    
    is_valid_type <- function(v, scalar=TRUE, vector=FALSE, matrix=FALSE, null=FALSE){
        if (scalar && data_type(v) == "SCALAR"){return(TRUE)}
        if (vector && data_type(v) == "VECTOR" && length(v) == vector){return(TRUE)}
        if (!isFALSE(matrix) && data_type(v) == "MATRIX" && all(dim(v) == matrix)){return(TRUE)}
        if (null && is.null(v)){return(TRUE)}
        return(FALSE)
    }
    
    params <- vector(mode="list", length=6)
    names(params) <- c("theta", "l0", "l1", "l2", "lows", "highs")
    params[[1]] <- theta
    params[[2]] <- l0
    params[[3]] <- l1
    params[[4]] <- l2
    params[[5]] <- lows
    params[[6]] <- highs
    
    if (identical(lows, -Inf) && identical(highs, +Inf)){
        params[['lows']] <- NULL
        params[['highs']] <- NULL
        lows <- NULL
        highs <- NULL
    }
    
    if (identical(l1, 0)){
        params[['l1']] <- NULL
        l1 <- NULL
    } 
    
    # for (param_name in names(params)){
    #     param <- params[[param_name]]
    #     if (!(gL0Learn.is.real_scalar(param) 
    #           || is.vector(param)
    #           || is.matrix(param))){
    #         stop(sprintf("expected %s to be a scalar, vector, or matrix but isn't", param_name))
    #     }
    # }
    return_as_vector = FALSE
    
    if (data_type(params[['theta']]) == "SCALAR"){
        for (param_name in names(params)){
            param <- params[[param_name]]
            if (!is_valid_type(param, scalar=TRUE)){
                stop(sprintf("expected %s to be a scalar, but isn't", param_name))
            }
        }
    } else {
        if (data_type(params[['theta']]) == "VECTOR"){
            return_as_vector = TRUE
            theta_len = length(theta)
            for (param_name in names(params)){
                param <- params[[param_name]]
                if (!is_valid_type(param, scalar=TRUE, vector=theta_len, matrix = FALSE, null=FALSE)){
                    stop(sprintf("expected %s to be a scalar or vector of the same length as theta, but isn't", param_name))
                }
                # if (value_type(param) == "VECTOR" && length(param) != theta_len){
                #     stop(sprintf("expected %s to be a vector of the same length as theta, but isn't", param_name))
                # } else if (value_type(param) == "MATRIX"){
                #     stop(sprintf("expected %s to be a scalar or vector of the same length as theta, but isn't", param_name))
                # }
            }
        }
        
        if (data_type(params[['theta']]) == "MATRIX"){
            theta_dim = dim(theta)
            for (param_name in names(params)){
                param <- params[[param_name]]
                if (!is_valid_type(param, scalar=TRUE, vector=FALSE, matrix =theta_dim, null=FALSE)){
                    stop(sprintf("expected %s to be a scalar or matrix of the same dims as theta, but isn't", param_name))
                }
                # if (value_type(param) == "MATRIX" && !identical(dim(param), theta_dim)){
                #     stop(sprintf("expected %s to be a matrix of the same dims as theta, but isn't", param_name))
                # } else if (value_type(param) == "VECTOR"){
                #     stop(sprintf("expected %s to be a scalar or matrix of the same dims as theta, but isn't", param_name))
                # }
            }
        }
    }
    
    if (!is_same_type(l0, l2)){
        stop("expected l0 and l2 to be the same type, but aren't")
    }
    
    if (!is.null(l1) && !is_same_type(l0, l1)){
        stop("expected l0 and l1 to be the same type, but aren't")
    }
    
    if (!is.null(lows) && !is.null(highs) && !is_same_type(lows, highs)){
        stop("expected lows and highs to be the same type, but aren't")
    }
        
    oracle <- test_Oracle_prox(theta, l0, l1, l2, lows, highs)
    
    if (return_as_vector){
        return(oracle[,])
    } else{
        return(oracle)
    }
    
}
