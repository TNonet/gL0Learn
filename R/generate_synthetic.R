#' @import Matrix
#' 
#' @title Generate a synthetic data set for gL0Learn
#' 
#' @description Computes the ...
#' @param n The number of observations to generated. 
#' This will create a data matrix of shape (n, p)
#' @param p The number of features to generated. 
#' This will create a data matrix of shape (n, p)
#' @param model The method for generating datasets.
#' Currently only AR1 is supported
#' @param normalize The method for normailzing data
#' Currently only "covariance" is supported 
#' @param seed A seed to a random number generated
#' 
#' @export
gL0Learn.generate_synthetic <- function(n, p, model, normalize, seed = 1, ...){
    if (model == "KR1"){
        args = as.list(match.call())
        if (is.null(args$rho)){
            stop("When model is 'KR1', an additional parameter rho is needed." )
        }
        return(gL0Learn.generate_Toeplitz_correlation(n,
                                                      p,
                                                      rho=args$rho,
                                                      normalize=normalize,
                                                      seed=seed))
    } else {
        stop("model is not supported.")
    }
}