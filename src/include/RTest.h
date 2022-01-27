#ifndef RTEST_H
#define RTEST_H

#include <tuple>
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


/*
 * Theta Type: Double, Vec, Mat 
 * Penalty Type: Double, Mat
 * Bounds Type: Double, Mat
 * Access Type: point (i, j), rol/col (i), all ()
 * 
 * Total types:
 *  3*2*2*3 -> 36....
 *  
 * Then we need to try all 8 Bounds...
 * L0, L1, L2, L0L1, L0L2, L1L2, L0L1L2
 * 
 * 
 * That being said. We can reduce the 36 states to just one
 * If we know we have a working set up for:
 *  Theta Type: Mat
 *  Penalty Type: Mat
 *  Bounds Type: Mat
 *  Access Type: all ()
 * Then we can have some code automatically make sure the sub-options are valid.
 */

using theta_mapping = std::map<std::string, arma::mat>;

template <template<class, class> class O, class P, class B>
std::tuple<theta_mapping, theta_mapping> test_oracle(const arma::mat& theta,
                                                     const O<P, B>& o){
    /*
     *  Expected Type Relations :
     *      Case 1: o.prox(double) -> double
     *      Case 2: o.prox(double, ij_row_col) -> double
     *      Case 3: o.prox(double, i, j) -> double
     *      Case 4: o.prox(mat) -> mat
     *      Case 5: o.prox(mat, ij_row_col) -> vec
     *      Case 6: o.prox(mat, i, j) -> double
     *      
     *  Oracle Options:
     *      A: o(mat, mat); All cases should match!
     *          1: o.prox(double, i, j) -> double
     *          2: o.prox(mat) -> mat
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      B: o(mat, double);
     *          1: o.prox(double, i, j) -> double
     *          2: o.prox(mat) -> mat
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      C: o(double, mat);
     *          1. o.prox(double, i, j) -> double
     *          2: o.prox(mat) -> mat
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      D: o(double, double);
     *          1. o.prox(double, i, j) -> double
     *          2: o.prox(double) -> double
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      X: o(mat, NoBounds);
     *          1: o.prox(double, i, j) -> double
     *          2: o.prox(mat) -> mat
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      Y: o(double, NoBounds);
     *          1. o.prox(double, i, j) -> double
     *          2: o.prox(mat) -> mat
     *          2a:o.prox(double) -> double
     *          3: o.prox(mat, ij_row_col) -> vec
     *          4: o.prox(mat, i, j) -> double
     *      
     *  This should be true for Penalty type X:
     *      L0, L1, L2, L0L1, L0L2, L1L2, L0L1L2
     *  
     */
    
    const arma::uword p = theta.n_cols;
    
    // Oracle Option A:
    arma::mat theta_A_1(p, p);
    arma::mat theta_A_2(p, p);
    arma::mat theta_A_3(p, p);
    arma::mat theta_A_4(p, p);
    arma::mat theta_B_1(p, p);
    arma::mat theta_B_2(p, p);
    arma::mat theta_B_3(p, p);
    arma::mat theta_B_4(p, p);
    arma::mat theta_C_1(p, p);
    arma::mat theta_C_2(p, p);
    arma::mat theta_C_3(p, p);
    arma::mat theta_C_4(p, p);
    arma::mat theta_D_1(p, p);
    arma::mat theta_D_2(p, p);
    arma::mat theta_D_3(p, p);
    arma::mat theta_D_4(p, p);
    
    arma::mat theta_X_1(p, p);
    arma::mat theta_X_2(p, p);
    arma::mat theta_X_3(p, p);
    arma::mat theta_X_4(p, p);
    
    arma::mat theta_Y_1(p, p);
    arma::mat theta_Y_2(p, p);
    arma::mat theta_Y_2a(p, p);
    arma::mat theta_Y_3(p, p);
    arma::mat theta_Y_4(p, p);
    
    const auto o_NBs = O<P, NoBounds>(o.penalty, NoBounds());
    
    theta_A_2 = o.prox(theta);
    theta_X_2 = o_NBs.prox(theta);
    
    for (auto i = 0; i < p; ++i){
        theta_A_3.col(i) = o.prox(theta, i);
        theta_X_3.col(i) = o_NBs.prox(theta, i);
        for (auto j = 0; j < p; ++j){
            theta_A_1(i, j) = o.prox(theta, i, j);
            theta_X_1(i, j) = o_NBs.prox(theta, i, j);
            theta_A_4(i, j) = o.prox(theta(i, j), i, j);
            theta_X_4(i, j) = o_NBs.prox(theta(i, j), i, j);
            
            const decltype(o.bounds(i, j)) Bd = o.bounds(i, j);
            const auto o_Bd = O<P, decltype(o.bounds(i, j))>(o.penalty, Bd);
            
            theta_B_1(i, j) = o_Bd.prox(theta, i, j);
            theta_B_2(i, j) = o_Bd.prox(theta)(i, j);
            theta_B_3(i, j) = o_Bd.prox(theta, i)(j);
            theta_B_4(i, j) = o_Bd.prox(theta(i, j), i, j);
            
            const decltype(o.penalty(i, j)) Pd = o.penalty(i, j);
            const auto o_Pd = O<decltype(o.penalty(i, j)), B>(Pd, o.bounds);
            const auto o_Pd_NBs = O<decltype(o.penalty(i, j)), NoBounds>(Pd, NoBounds());
            
            theta_C_1(i, j) = o_Pd.prox(theta, i, j);
            theta_C_2(i, j) = o_Pd.prox(theta)(i, j);
            theta_C_3(i, j) = o_Pd.prox(theta, i)(j);
            theta_C_4(i, j) = o_Pd.prox(theta(i, j), i, j);
            
            theta_Y_1(i, j) = o_Pd_NBs.prox(theta, i, j);
            theta_Y_2(i, j) = o_Pd_NBs.prox(theta)(i, j);
            theta_Y_2a(i, j) = o_Pd_NBs.prox(theta(i ,j));
            theta_Y_3(i, j) = o_Pd_NBs.prox(theta, i)(j);
            theta_Y_4(i, j) = o_Pd_NBs.prox(theta(i, j), i, j);
            
            const auto o_Pd_Bd = O<decltype(o.penalty(i, j)), decltype(o.bounds(i, j))>(Pd, Bd);
            
            theta_D_1(i, j) = o_Pd_Bd.prox(theta, i, j);
            theta_D_2(i, j) = o_Pd_Bd.prox(theta(i, j));
            theta_D_3(i, j) = o_Pd_Bd.prox(theta, i)(j);
            theta_D_4(i, j) = o_Pd_Bd.prox(theta(i, j), i, j);
            
            
        }
    }
    
    std::map< std::string, arma::mat> results;
    results["theta_A_1"] = theta_A_1;
    results["theta_A_2"] = theta_A_2;
    results["theta_A_3"] = theta_A_3;
    results["theta_A_4"] = theta_A_4;
    results["theta_B_1"] = theta_B_1;
    results["theta_B_2"] = theta_B_2;
    results["theta_B_3"] = theta_B_3;
    results["theta_B_4"] = theta_B_4;
    results["theta_C_1"] = theta_C_1;
    results["theta_C_2"] = theta_C_2;
    results["theta_C_3"] = theta_C_3;
    results["theta_C_4"] = theta_C_4;
    results["theta_D_1"] = theta_C_1;
    results["theta_D_2"] = theta_C_2;
    results["theta_D_3"] = theta_C_3;
    results["theta_D_4"] = theta_C_4;
    
    std::map< std::string, arma::mat> results_NBs;
    results_NBs["theta_X_1"] = theta_X_1;
    results_NBs["theta_X_2"] = theta_X_2;
    results_NBs["theta_X_3"] = theta_X_3;
    results_NBs["theta_X_4"] = theta_X_4;
    results_NBs["theta_Y_1"] = theta_Y_1;
    results_NBs["theta_Y_2"] = theta_Y_2;
    results_NBs["theta_Y_2a"] = theta_Y_2a;
    results_NBs["theta_Y_3"] = theta_Y_3;
    results_NBs["theta_Y_4"] = theta_Y_4;
    
    
    return {results, results_NBs};
}


#endif // RTEST_H