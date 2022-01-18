#include "RgL0Learn.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
bool is_sympd(const arma::mat& x){
    return x.is_sympd();
}

// [[Rcpp::export]]
arma::umat upper_triangluar_coords(const arma::uword p){
    return coordinate_matrix_from_vector(upper_triangle_coordinate_vector(p));
}

// [[Rcpp::export]]
bool check_coordinate_matrix_is_valid(const arma::umat& coords_ma,
                                      const bool for_order = true,
                                      const bool for_upper_triangle = true){
    return check_coordinate_matrix(coords_ma, for_order, for_upper_triangle);
}

// [[Rcpp::export]]
bool check_is_valid_coordinate_subset(const arma::umat& larger_coord_set,
                                      const arma::umat& smaller_coord_set){
    return check_is_coordinate_subset(larger_coord_set,smaller_coord_set);
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
                          const arma::umat& initial_active_set,
                          const arma::umat& super_active_set,
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
                                      initial_active_set,
                                      super_active_set,
                                      atol, 
                                      rtol, 
                                      max_iter);
    
    return(Rcpp::List::create(Rcpp::Named("theta") = l.theta,
                              Rcpp::Named("R") = l.R,
                              Rcpp::Named("costs") = l.costs,
                              Rcpp::Named("active_set_size") = l.active_set_size));
}
