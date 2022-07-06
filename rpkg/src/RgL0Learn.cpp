#include "RgL0Learn.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
bool is_sympd(const arma::mat &x) { return x.is_sympd(); }

// [[Rcpp::export]]
arma::umat upper_triangluar_coords(const arma::uword p) {
  return coordinate_matrix_from_vector(upper_triangle_coordinate_vector(p));
}

// [[Rcpp::export]]
bool check_coordinate_matrix_is_valid(const arma::umat &coords_ma,
                                      const bool for_order = true,
                                      const bool for_upper_triangle = true) {
  return check_coordinate_matrix(coords_ma, for_order, for_upper_triangle);
}

// [[Rcpp::export]]
bool check_is_valid_coordinate_subset(const arma::umat &larger_coord_set,
                                      const arma::umat &smaller_coord_set) {
  return check_is_coordinate_subset(larger_coord_set, smaller_coord_set);
}
