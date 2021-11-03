#' @import Matrix
#' 
#' @title Generate a independent correlated data set for gL0Learn
#' 
#' @description Computes the ...
#' @param n The number of observations to generated. 
#' This will create a data matrix of shape (n, p)
#' @param p The number of features to generated. 
#' This will create a data matrix of shape (n, p).
#' @param normalize The method for normalizing data
#' Currently only "covariance" is supported 
#' @param seed A seed to a random number generated
#' 
#' @export
gL0Learn.generate_independent <- function(n, p, normalize, seed = 1){
    set.seed(seed)
    X <- matrix(rnorm(n*p), n, p)
    sigma = diag(p)
    theta = diag(p)
    
    if (normalize == "covariance"){
        # Nothing
    } else {
        # Nothing also
    }
    return(list(X=X, sigma=sigma, theta=theta))
        
    
}