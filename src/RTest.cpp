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

// [[Rcpp::export]]
Rcpp::List test_oracle_L0(const arma::mat& theta,
                          const arma::mat& l0,
                          const arma::mat& l1,
                          const arma::mat& l2,
                          const arma::mat& lows,
                          const arma::mat& highs){
    
    const Bounds<arma::mat> b_m(lows, highs);
    const PenaltyL0<arma::mat> p_m(l0);
    const Oracle<PenaltyL0<arma::mat>, Bounds<arma::mat>> o_m_m(p_m, b_m);
    
    const auto results = test_oracle(theta, o_m_m);
    
    const SEXP results_0 = Rcpp::wrap(std::get<0>(results));
    const SEXP results_1 = Rcpp::wrap(std::get<1>(results));
    return Rcpp::List::create(Rcpp::Named("with_bounds") = results_0,
                              Rcpp::Named("without_bounds") = results_1);
}

// [[Rcpp::export]]
Rcpp::List test_oracle_L0L2(const arma::mat& theta,
                            const arma::mat& l0,
                            const arma::mat& l1,
                            const arma::mat& l2,
                            const arma::mat& lows,
                            const arma::mat& highs){
    
    const Bounds<arma::mat> b_m(lows, highs);
    const PenaltyL0L2<arma::mat> p_m(l0, l2);
    const Oracle<PenaltyL0L2<arma::mat>, Bounds<arma::mat>> o_m_m(p_m, b_m);
    
    const auto results = test_oracle(theta, o_m_m);
    
    const SEXP results_0 = Rcpp::wrap(std::get<0>(results));
    const SEXP results_1 = Rcpp::wrap(std::get<1>(results));
    return Rcpp::List::create(Rcpp::Named("with_bounds") = results_0,
                              Rcpp::Named("without_bounds") = results_1);
}

// [[Rcpp::export]]
Rcpp::List test_oracle_L0L1L2(const arma::mat& theta,
                              const arma::mat& l0,
                              const arma::mat& l1,
                              const arma::mat& l2,
                              const arma::mat& lows,
                              const arma::mat& highs){
    
    const Bounds<arma::mat> b_m(lows, highs);
    const PenaltyL0L1L2<arma::mat> p_m(l0, l1, l2);
    const Oracle<PenaltyL0L1L2<arma::mat>, Bounds<arma::mat>> o_m_m(p_m, b_m);
    
    const auto results = test_oracle(theta, o_m_m);
    
    const SEXP results_0 = Rcpp::wrap(std::get<0>(results));
    const SEXP results_1 = Rcpp::wrap(std::get<1>(results));
    return Rcpp::List::create(Rcpp::Named("with_bounds") = results_0,
                              Rcpp::Named("without_bounds") = results_1);
}

// [[Rcpp::export]]
SEXP test_union_of_correlated_features(const arma::mat & x,
                                       const double threshold){
    
    const coordinate_vector cv = union_of_correlated_features(x, threshold);
    
    Rcpp::Rcout << cv;
    
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
arma::umat test_union_of_correlated_features2(const arma::mat & x,
                                              const double threshold){
    return union_of_correlated_features2(x, threshold);
}

// [[Rcpp::export]]
std::vector<int> test_sorted_vector_difference(const std::vector<int> larger,
                                   const std::vector<int> smaller){
    
    return sorted_vector_difference(larger, smaller);
}

// [[Rcpp::export]]
std::vector<int> test_sorted_vector_difference2(const std::vector<int> larger,
                                               const std::vector<int> smaller){
    
    return sorted_vector_difference2(larger, smaller);
}

// [[Rcpp::export]]
std::vector<int> test_insert_sorted_vector_into_sorted_vector(const std::vector<int> x1,
                                                              const std::vector<int> x2){
    
    return insert_sorted_vector_into_sorted_vector(x1, x2);
}

// [[Rcpp::export]]
arma::umat test_coordinate_matrix_to_vector_to_matrix(const arma::umat& coords_ma){
    auto coords_vec = coordinate_vector_from_matrix(coords_ma);
    return coordinate_matrix_from_vector(coords_vec);

}

// [[Rcpp::export]]
arma::umat test_unravel_indices(const arma::uvec& indices,
                                const arma::uword p){
    return unravel_ut_indices(indices, p);
}
