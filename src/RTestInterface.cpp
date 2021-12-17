// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "RcppArmadillo.h"
#include "oracle2.h"
// [[Rcpp::depends(RcppArmadillo)]]

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

// [[Rcpp::export]]
double test_OracleScalarL0L2NoBounds_prox_double(const double theta,
                                                 const double l0, 
                                                 const double l2){
    ScalarPenaltyL0L2 p(l0, l2);
    OracleScalarL0L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleScalarL0L2NoBounds_prox_vec(const arma::vec theta,
                                                 const double l0, 
                                                 const double l2){
    ScalarPenaltyL0L2 p(l0, l2);
    OracleScalarL0L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L2NoBounds_prox_vec(const arma::vec theta,
                                                 const arma::vec l0,
                                                 const arma::vec l2){
    VectorPenaltyL0L2 p(l0, l2);
    OracleVectorL0L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// [[Rcpp::export]]
double test_OracleScalarL0L1L2NoBounds_prox_double(const double theta, 
                                                   const double l0, 
                                                   const double l1, 
                                                   const double l2){
    ScalarPenaltyL0L1L2 p(l0, l1, l2);
    OracleScalarL0L1L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleScalarL0L1L2NoBounds_prox_vec(const arma::vec theta, 
                                                   const double l0, 
                                                   const double l1, 
                                                   const double l2){
    ScalarPenaltyL0L1L2 p(l0, l1, l2);
    OracleScalarL0L1L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L1L2NoBounds_prox_vec(const arma::vec theta,
                                                   const arma::vec l0, 
                                                   const arma::vec l1, 
                                                   const arma::vec l2){
    VectorPenaltyL0L1L2 p(l0, l1, l2);
    OracleVectorL0L1L2NoBounds o(p, NoBounds());
    return o.prox(theta);
}

// With ScalarBounds

// [[Rcpp::export]]
double test_OracleScalarL0L2ScalarBounds_prox_double(const double theta,
                                                     const double l0,
                                                     const double l2, 
                                                     const double lows,
                                                     const double highs){
    ScalarPenaltyL0L2 p(l0, l2);
    ScalarBounds b(lows, highs);
    OracleScalarL0L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleScalarL0L2ScalarBounds_prox_vec(const arma::vec theta,
                                                     const double l0,
                                                     const double l2,
                                                     const double lows,
                                                     const double highs){
    ScalarPenaltyL0L2 p(l0, l2);
    ScalarBounds b(lows, highs);
    OracleScalarL0L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L2ScalarBounds_prox_vec(const arma::vec theta,
                                                     const arma::vec l0,
                                                     const arma::vec l2,
                                                     const double lows,
                                                     const double highs){
    VectorPenaltyL0L2 p(l0, l2);
    ScalarBounds b(lows, highs);
    OracleVectorL0L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
double test_OracleScalarL0L1L2ScalarBounds_prox_double(const double theta,
                                                       const double l0,
                                                       const double l1,
                                                       const double l2,
                                                       const double lows,
                                                       const double highs){
    ScalarPenaltyL0L1L2 p(l0, l1, l2);
    ScalarBounds b(lows, highs);
    OracleScalarL0L1L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleScalarL0L1L2ScalarBounds_prox_vec(const arma::vec theta,
                                                       const double l0,
                                                       const double l1,
                                                       const double l2,
                                                       const double lows,
                                                       const double highs){
    ScalarPenaltyL0L1L2 p(l0, l1, l2);
    ScalarBounds b(lows, highs);
    OracleScalarL0L1L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L1L2ScalarBounds_prox_vec(const arma::vec theta,
                                                       const arma::vec l0,
                                                       const arma::vec l1,
                                                       const arma::vec l2,
                                                       const double lows,
                                                       const double highs){
    VectorPenaltyL0L1L2 p(l0, l1, l2);
    ScalarBounds b(lows, highs);
    OracleVectorL0L1L2ScalarBounds o(p, b);
    return o.prox(theta);
}

// With VectorBounds

// [[Rcpp::export]]
arma::vec test_OracleScalarL0L2VectorBounds_prox_vec(const arma::vec theta,
                                                     const double l0,
                                                     const double l2,
                                                     const arma::vec lows,
                                                     const arma::vec highs){
    ScalarPenaltyL0L2 p(l0, l2);
    VectorBounds b(lows, highs);
    OracleScalarL0L2VectorBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L2VectorBounds_prox_vec(const arma::vec theta,
                                                     const arma::vec l0,
                                                     const arma::vec l2,
                                                     const arma::vec lows,
                                                     const arma::vec highs){
    VectorPenaltyL0L2 p(l0, l2);
    VectorBounds b(lows, highs);
    OracleVectorL0L2VectorBounds o(p, b);
    return o.prox(theta);
}


// [[Rcpp::export]]
arma::vec test_OracleScalarL0L1L2VectorBounds_prox_vec(const arma::vec theta,
                                                       const double l0,
                                                       const double l1,
                                                       const double l2,
                                                       const arma::vec lows,
                                                       const arma::vec highs){
    ScalarPenaltyL0L1L2 p(l0, l1, l2);
    VectorBounds b(lows, highs);
    OracleScalarL0L1L2VectorBounds o(p, b);
    return o.prox(theta);
}

// [[Rcpp::export]]
arma::vec test_OracleVectorL0L1L2VectorBounds_prox_vec(const arma::vec theta,
                                                       const arma::vec l0,
                                                       const arma::vec l1,
                                                       const arma::vec l2,
                                                       const arma::vec lows,
                                                       const arma::vec highs){
    VectorPenaltyL0L1L2 p(l0, l1, l2);
    VectorBounds b(lows, highs);
    OracleVectorL0L1L2VectorBounds o(p, b);
    return o.prox(theta);
}
