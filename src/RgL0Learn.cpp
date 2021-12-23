#include "RgL0Learn.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
bool is_sympd(const arma::mat& x){
    return x.is_sympd();
}


// [[Rcpp::export]]
Rcpp::List gL0Learn_fit_R(const arma::mat& Y,
                          const arma::mat& theta_init,
                          const SEXP l0,
                          const SEXP l1,
                          const SEXP l2,
                          const std::string algorithm,
                          const SEXP lows,
                          const SEXP highs,
                          const double atol,
                          const double rtol,
                          const size_t max_iter){
    
    const fitmodel l = gL0Learn_fit_C(Y,
                                      theta_init,
                                      l0,
                                      l1,
                                      l2, 
                                      algorithm, 
                                      lows, 
                                      highs, 
                                      atol, 
                                      rtol, 
                                      max_iter);
    
    return(Rcpp::List::create(Rcpp::Named("theta") = l.theta,
                              Rcpp::Named("R") = l.R));
}
