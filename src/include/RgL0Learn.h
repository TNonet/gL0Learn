#ifndef RINTERFACE_H
#define RINTERFACE_H
#include <algorithm>
#include "RcppArmadillo.h"
#include "gL0Learn.h"
#include "oracle.h"
#include "utils.h"

template <template <class> class P, class E>
fitmodel gL0Learn_fit_sub_penalty(const arma::mat& Y,
                                  const arma::mat& theta_init,
                                  const P<E>& penalty,
                                  const std::string algorithm,
                                  const SEXP& lows,
                                  const SEXP& highs,
                                  const double atol,
                                  const double rtol,
                                  const size_t max_iter) {
    if (Rf_isNull(lows) && Rf_isNull(highs)){
        return gL0LearnFit(Y, theta_init,
                           penalty,
                           NoBounds(),
                           algorithm, atol, rtol, max_iter);
    } else {
        if (is_double_SEXP(lows) && is_double_SEXP(highs)){
            return gL0LearnFit(Y, theta_init,
                               penalty,
                               Bounds<double>(Rcpp::as<double>(lows), Rcpp::as<double>(highs)),
                               algorithm, atol, rtol, max_iter);
        } else {
            return gL0LearnFit(Y, theta_init,
                               penalty,
                               Bounds<arma::mat>(Rcpp::as<arma::mat>(lows), Rcpp::as<arma::mat>(highs)),
                               algorithm, atol, rtol, max_iter);
        }
        
    }
    
}


fitmodel gL0Learn_fit_C(const arma::mat& Y,
                        const arma::mat& theta_init,
                        const SEXP& l0,
                        const SEXP& l1,
                        const SEXP& l2,
                        const std::string algorithm,
                        const SEXP& lows,
                        const SEXP& highs,
                        const double atol,
                        const double rtol,
                        const size_t max_iter) {
    
    if (Rf_isNull(l1)){
        if (is_double_SEXP(l0) && is_double_SEXP(l2)){
            return gL0Learn_fit_sub_penalty(Y, theta_init,
                                            PenaltyL0L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l2)),
                                            algorithm, lows, highs, atol, rtol, max_iter);
        } else {
            return gL0Learn_fit_sub_penalty(Y, theta_init,
                                            PenaltyL0L2<arma::mat>(Rcpp::as<arma::mat>(l0), Rcpp::as<arma::mat>(l2)),
                                            algorithm, lows, highs, atol, rtol, max_iter);
        }
    } else {
        if (is_double_SEXP(l0) && is_double_SEXP(l1) && is_double_SEXP(l2)){
            return gL0Learn_fit_sub_penalty(Y,theta_init,
                                            PenaltyL0L1L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l1), Rcpp::as<double>(l2)),
                                            algorithm, lows, highs, atol, rtol, max_iter);
        } else {
            return gL0Learn_fit_sub_penalty(Y, theta_init,
                                            PenaltyL0L1L2<arma::mat>(Rcpp::as<arma::mat>(l0), Rcpp::as<arma::mat>(l1), Rcpp::as<arma::mat>(l2)),
                                            algorithm, lows, highs, atol, rtol, max_iter);
        }
    }
}





#endif