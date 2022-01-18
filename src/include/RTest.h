#ifndef RTEST_H
#define RTEST_H

#include "RcppArmadillo.h"
#include "oracle.h"
#include "utils.h"
#include "active_set.h"

typedef Bounds<double> ScalarBounds;
typedef Bounds<arma::vec> VectorBounds;

// typedef PenaltyL0<double> ScalarPenaltyL0;
// typedef PenaltyL0<arma::vec> VectorPenaltyL0;
typedef PenaltyL0L2<double> ScalarPenaltyL0L2;
typedef PenaltyL0L2<arma::vec> VectorPenaltyL0L2;
typedef PenaltyL0L1L2<double> ScalarPenaltyL0L1L2;
typedef PenaltyL0L1L2<arma::vec> VectorPenaltyL0L1L2;

typedef Oracle<ScalarPenaltyL0L2, NoBounds>       OracleScalarL0L2NoBounds;
typedef Oracle<VectorPenaltyL0L2, NoBounds>       OracleVectorL0L2NoBounds;
typedef Oracle<ScalarPenaltyL0L1L2, NoBounds>     OracleScalarL0L1L2NoBounds;
typedef Oracle<VectorPenaltyL0L1L2, NoBounds>     OracleVectorL0L1L2NoBounds;
typedef Oracle<ScalarPenaltyL0L2, ScalarBounds>   OracleScalarL0L2ScalarBounds;
typedef Oracle<VectorPenaltyL0L2, ScalarBounds>   OracleVectorL0L2ScalarBounds;
typedef Oracle<ScalarPenaltyL0L1L2, ScalarBounds> OracleScalarL0L1L2ScalarBounds;
typedef Oracle<VectorPenaltyL0L1L2, ScalarBounds> OracleVectorL0L1L2ScalarBounds;
typedef Oracle<ScalarPenaltyL0L2, VectorBounds>   OracleScalarL0L2VectorBounds;
typedef Oracle<VectorPenaltyL0L2, VectorBounds>   OracleVectorL0L2VectorBounds;
typedef Oracle<ScalarPenaltyL0L1L2, VectorBounds> OracleScalarL0L1L2VectorBounds;
typedef Oracle<VectorPenaltyL0L1L2, VectorBounds> OracleVectorL0L1L2VectorBounds;

template <template <class> class P, class E>
double Oracle_prox_sub_penalty(const double& theta,
                               const P<E>& penalty,
                               const SEXP& lows,
                               const SEXP& highs){
    
    if (Rf_isNull(lows) && Rf_isNull(highs)){
        return Oracle<P<E>, NoBounds>(penalty, NoBounds()).prox(theta);
    } else {
        if (is_double_SEXP(lows) && is_double_SEXP(highs)){
            return Oracle<P<E>, Bounds<double>>(penalty, Bounds<double>(Rcpp::as<double>(lows), Rcpp::as<double>(highs))).prox(theta);
        } else {
            Rcpp::stop("Must be same types");
        }
        
    }    
}

template <template <class> class P, class E>
arma::vec Oracle_prox_sub_penalty(const arma::vec& theta,
                                  const P<E>& penalty,
                                  const SEXP& lows,
                                  const SEXP& highs){
    
    if (Rf_isNull(lows) && Rf_isNull(highs)){
        return Oracle<P<E>, NoBounds>(penalty, NoBounds()).prox(theta);
    } else {
        if (is_double_SEXP(lows) && is_double_SEXP(highs)){
            return Oracle<P<E>, Bounds<double>>(penalty, Bounds<double>(Rcpp::as<double>(lows), Rcpp::as<double>(highs))).prox(theta);
        } else if (Rf_isVector(lows) && Rf_isVector(highs)){
            return Oracle<P<E>, Bounds<arma::vec>>(penalty, Bounds<arma::vec>(Rcpp::as<arma::vec>(lows), Rcpp::as<arma::vec>(highs))).prox(theta);
        } else {
            Rcpp::stop("Must be same types");
        }
        
    }    
}

template <template <class> class P, class E>
arma::mat Oracle_prox_sub_penalty(const arma::mat& theta,
                                  const P<E>& penalty,
                                  const SEXP& lows,
                                  const SEXP& highs){
    
    if (Rf_isNull(lows) && Rf_isNull(highs)){
        return Oracle<P<E>, NoBounds>(penalty, NoBounds()).prox(theta);
    } else {
        if (is_double_SEXP(lows) && is_double_SEXP(highs)){
            return Oracle<P<E>, Bounds<double>>(penalty, Bounds<double>(Rcpp::as<double>(lows), Rcpp::as<double>(highs))).prox(theta);
        } else if (Rf_isMatrix(lows) && Rf_isMatrix(highs)){
            return Oracle<P<E>, Bounds<arma::mat>>(penalty, Bounds<arma::mat>(Rcpp::as<arma::mat>(lows), Rcpp::as<arma::mat>(highs))).prox(theta);
        } else if (Rf_isVector(lows) && Rf_isVector(highs)){
            return Oracle<P<E>, Bounds<arma::vec>>(penalty, Bounds<arma::vec>(Rcpp::as<arma::vec>(lows), Rcpp::as<arma::vec>(highs))).prox(theta);
        } else {
            Rcpp::stop("Must be same types");
        }
        
    }    
}


double Oracle_prox_c(const double& theta,
                     const SEXP& l0,
                     const SEXP& l1,
                     const SEXP& l2,
                     const SEXP& lows,
                     const SEXP& highs){
    if (Rf_isNull(l1)){
        if (is_double_SEXP(l0) && is_double_SEXP(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    } else {
        if (is_double_SEXP(l0) && is_double_SEXP(l1) && is_double_SEXP(l1)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l1), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    }
}

arma::vec Oracle_prox_c(const arma::vec& theta, 
                        const SEXP& l0, 
                        const SEXP& l1, 
                        const SEXP& l2, 
                        const SEXP& lows, 
                        const SEXP& highs){
    if (Rf_isNull(l1)){
        if (is_double_SEXP(l0) && is_double_SEXP(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else if (Rf_isVector(l0) && Rf_isVector(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<arma::vec>(Rcpp::as<arma::vec>(l0), Rcpp::as<arma::vec>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    } else {
        if (is_double_SEXP(l0) && is_double_SEXP(l1) && is_double_SEXP(l1)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l1), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else if (Rf_isVector(l0) && Rf_isVector(l1) && Rf_isVector(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<arma::vec>(Rcpp::as<arma::vec>(l0), Rcpp::as<arma::vec>(l1), Rcpp::as<arma::vec>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    }
}

arma::mat Oracle_prox_c(const arma::mat& theta, 
                        const SEXP& l0, 
                        const SEXP& l1, 
                        const SEXP& l2, 
                        const SEXP& lows, 
                        const SEXP& highs){
    if (Rf_isNull(l1)){
        if (is_double_SEXP(l0) && is_double_SEXP(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else if (Rf_isMatrix(l0) && Rf_isMatrix(l2)) {
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<arma::mat>(Rcpp::as<arma::mat>(l0), Rcpp::as<arma::mat>(l2)),
                                           lows, highs);
        } else if (Rf_isVector(l0) && Rf_isVector(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L2<arma::vec>(Rcpp::as<arma::vec>(l0), Rcpp::as<arma::vec>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    } else {
        if (is_double_SEXP(l0) && is_double_SEXP(l1) && is_double_SEXP(l1)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<double>(Rcpp::as<double>(l0), Rcpp::as<double>(l1), Rcpp::as<double>(l2)),
                                           lows, highs);
        } else if (Rf_isMatrix(l0) &&  Rf_isMatrix(l1) && Rf_isMatrix(l2)) {
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<arma::mat>(Rcpp::as<arma::mat>(l0), Rcpp::as<arma::mat>(l1), Rcpp::as<arma::mat>(l2)),
                                           lows, highs);
        } else if (Rf_isVector(l0) && Rf_isVector(l1) && Rf_isVector(l2)){
            return Oracle_prox_sub_penalty(theta,
                                           PenaltyL0L1L2<arma::vec>(Rcpp::as<arma::vec>(l0), Rcpp::as<arma::vec>(l1), Rcpp::as<arma::vec>(l2)),
                                           lows, highs);
        } else {
            Rcpp::stop("Must be same types");
        }
    }
}


#endif // RTEST_H