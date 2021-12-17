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
gL0Learn.gfit <- function(x,
                          theta_init=NULL,
                          atol=1e-6,
                          rtol=1e-6,
                          l0=0,
                          l1=0,
                          l2=0,
                          max_iter=100,
                          algorithm="CD",
                          swap_iters=NULL,
                          scale_x=FALSE){
    
    x_dims = dim(x)
    if (length(x_dims) != 2){
        stop("L0Learn.gfit requires x to be a 2D array type")
    }
    n = x_dims[[1]]
    p = x_dims[[2]]
    
    y = NULL
    if (scale_x){
        y = x/sqrt(n)
    } else {
        y = x
    }
    
    if (is.null(theta_init)){
        theta_init = diag(p)
    }
    
    ## TODO Check the diagonals of theta_init are non-zero and non negative
    
    if (algorithm == "CD"){
        return(.Call('_gL0Learn_gL0Learn_fit', PACKAGE = 'gL0Learn', y, theta_init, atol, rtol, l0, l1, l2, max_iter))
    } else {
        return(.Call('_gL0Learn_gL0Learn_psifit', PACKAGE = 'gL0Learn', Y, theta_init, atol, rtol, l0, l1, l2, max_iter))
    }
    

}
    