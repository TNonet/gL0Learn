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
                          l0=0,
                          l1=0,
                          l2=0,
                          lows=-Inf,
                          highs=Inf,
                          max_iter=100,
                          algorithm="CD",
                          atol=1e-6,
                          rtol=1e-6,
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
    
    
   return(gL0Learn_fit_R(y, theta_init, l0, l1, l2, algorithm, lows, highs, atol, rtol, max_iter))
  
}
    