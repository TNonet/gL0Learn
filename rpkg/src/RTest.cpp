// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil;
// -*-
#include "RTest.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
SEXP test_union_of_correlated_features(const arma::mat &x,
                                       const double threshold) {
  const coordinate_vector cv = union_of_correlated_features(x, threshold);

  COUT << cv;

  Rcpp::List coordinate_list(cv.size());

  for (size_t i = 0; i < cv.size(); ++i) {
    Rcpp::List c(2);
    c[0] = std::get<0>(cv[i]);
    c[1] = std::get<1>(cv[i]);
    coordinate_list[i] = c;
  }

  return coordinate_list;
}

// [[Rcpp::export]]
arma::umat test_union_of_correlated_features2(const arma::mat &x,
                                              const double threshold) {
  return union_of_correlated_features2(x, threshold);
}

// [[Rcpp::export]]
arma::umat test_coordinate_matrix_to_vector_to_matrix(
    const arma::umat &coords_ma) {
  auto coords_vec = coordinate_vector_from_matrix(coords_ma);
  return coordinate_matrix_from_vector(coords_vec);
}

// [[Rcpp::export]]
arma::umat test_unravel_indices(const arma::uvec &indices,
                                const arma::uword p) {
  return unravel_ut_indices(indices, p);
}
