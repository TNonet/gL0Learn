// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "RTest.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
SEXP test_Oracle_prox(const SEXP &theta,
                      const SEXP &l0,
                      const SEXP &l1,
                      const SEXP &l2,
                      const SEXP &lows,
                      const SEXP &highs){
    
    if (is_double_SEXP(theta)){
        const double prox = Oracle_prox_c(Rcpp::as<double>(theta), l0, l1, l2, lows, highs);
        return Rcpp::wrap(prox);
    } else if (Rf_isMatrix(theta)){
        const arma::mat prox = Oracle_prox_c(Rcpp::as<arma::mat>(theta), l0, l1, l2, lows, highs);
        return Rcpp::wrap(prox);
    } else { // Assumed to be a vector
        const arma::vec prox = Oracle_prox_c(Rcpp::as<arma::vec>(theta), l0, l1, l2, lows, highs);
        return Rcpp::wrap(prox);
    }
}