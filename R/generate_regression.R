#' @import Matrix
#' 
#' @title Generate a regression data set for gL0Learn
#' 
#' @description Computes the ...
#' @param n The number of observations to generated. 
#' This will create a data matrix of shape (n, p)
#' @param p The number of features to generated. 
#' This will create a data matrix of shape (n, p)
#' @param rwo The correlation for ...
#' @param normalize The method for normalizing data
#' Currently only "covariance" is supported 
#' @param seed A seed to a random number generated
#' 
#' @export
gL0Learn.generate_regression <- function(n, p, k, val, normalize, seed = 1, ...){
    if (k > p-1){
        stop("k must be in less than or equal to p - 1")
    }
    set.seed(seed)
    X <- matrix(rnorm(n*(p-1)), n, p-1)
    beta <- rep(0, p-1)
    nnz_beta_ix <- sample(1:(p-1), k, replace = FALSE)
    nnz_beta_values <- (2*sample(0:1, k, replace = TRUE) - 1)*val
    beta <- replace(beta, nnz_beta_ix, nnz_beta_values)
    noise <- rnorm(n)
    y <- X%*%beta + noise
    X <- cbind(y, X)
    
    sigma <- diag(p)
    sigma[1, 1] <- 1 + sum(beta**2)
    sigma[2:p, 1] <- -beta
    sigma[1, 2:p] <- -beta
    
    theta <- diag(p)
    beta_banded <- replicate(p-1, beta)
    theta[2:p, 2:p] <- beta_banded%*%t(beta_banded)
    theta[1, 2:p] <- beta
    theta[2:p, 1] <- beta
    
    if (normalize == "covariance"){
        tiled_sigma_diag <- replicate(n,sqrt(diag(sigma)))
        
        X <- X*tiled_sigma_diag
        sigma <- sigma*tiled_sigma_diag*t(tiled_sigma_diag)
        theta <- theta/tiled_sigma_diag/t(tiled_sigma_diag)
    } else {
        tiled_theta_diag <- replicate(n,sqrt(diag(theta)))
        
        X <- X*tiled_theta_diag
        sigma <- sigma*tiled_theta_diag*t(tiled_theta_diag)
        theta <- theta/tiled_theta_diag/t(tiled_theta_diag)
    }
    return(list(X=X, sigma=sigma, theta=theta))
        
    
}