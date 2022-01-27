// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// test_Oracle_prox
SEXP test_Oracle_prox(const SEXP& theta, const SEXP& l0, const SEXP& l1, const SEXP& l2, const SEXP& lows, const SEXP& highs);
RcppExport SEXP _gL0Learn_test_Oracle_prox(SEXP thetaSEXP, SEXP l0SEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP lowsSEXP, SEXP highsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const SEXP& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const SEXP& >::type l0(l0SEXP);
    Rcpp::traits::input_parameter< const SEXP& >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< const SEXP& >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< const SEXP& >::type lows(lowsSEXP);
    Rcpp::traits::input_parameter< const SEXP& >::type highs(highsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_Oracle_prox(theta, l0, l1, l2, lows, highs));
    return rcpp_result_gen;
END_RCPP
}
// test_oracle_L0
Rcpp::List test_oracle_L0(const arma::mat& theta, const arma::mat& l0, const arma::mat& l1, const arma::mat& l2, const arma::mat& lows, const arma::mat& highs);
RcppExport SEXP _gL0Learn_test_oracle_L0(SEXP thetaSEXP, SEXP l0SEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP lowsSEXP, SEXP highsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l0(l0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type lows(lowsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type highs(highsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_oracle_L0(theta, l0, l1, l2, lows, highs));
    return rcpp_result_gen;
END_RCPP
}
// test_oracle_L0L2
Rcpp::List test_oracle_L0L2(const arma::mat& theta, const arma::mat& l0, const arma::mat& l1, const arma::mat& l2, const arma::mat& lows, const arma::mat& highs);
RcppExport SEXP _gL0Learn_test_oracle_L0L2(SEXP thetaSEXP, SEXP l0SEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP lowsSEXP, SEXP highsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l0(l0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type lows(lowsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type highs(highsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_oracle_L0L2(theta, l0, l1, l2, lows, highs));
    return rcpp_result_gen;
END_RCPP
}
// test_oracle_L0L1L2
Rcpp::List test_oracle_L0L1L2(const arma::mat& theta, const arma::mat& l0, const arma::mat& l1, const arma::mat& l2, const arma::mat& lows, const arma::mat& highs);
RcppExport SEXP _gL0Learn_test_oracle_L0L1L2(SEXP thetaSEXP, SEXP l0SEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP lowsSEXP, SEXP highsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l0(l0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type lows(lowsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type highs(highsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_oracle_L0L1L2(theta, l0, l1, l2, lows, highs));
    return rcpp_result_gen;
END_RCPP
}
// test_union_of_correlated_features
SEXP test_union_of_correlated_features(const arma::mat& x, const double threshold);
RcppExport SEXP _gL0Learn_test_union_of_correlated_features(SEXP xSEXP, SEXP thresholdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double >::type threshold(thresholdSEXP);
    rcpp_result_gen = Rcpp::wrap(test_union_of_correlated_features(x, threshold));
    return rcpp_result_gen;
END_RCPP
}
// test_union_of_correlated_features2
arma::umat test_union_of_correlated_features2(const arma::mat& x, const double threshold);
RcppExport SEXP _gL0Learn_test_union_of_correlated_features2(SEXP xSEXP, SEXP thresholdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double >::type threshold(thresholdSEXP);
    rcpp_result_gen = Rcpp::wrap(test_union_of_correlated_features2(x, threshold));
    return rcpp_result_gen;
END_RCPP
}
// test_sorted_vector_difference
std::vector<int> test_sorted_vector_difference(const std::vector<int> larger, const std::vector<int> smaller);
RcppExport SEXP _gL0Learn_test_sorted_vector_difference(SEXP largerSEXP, SEXP smallerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type larger(largerSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type smaller(smallerSEXP);
    rcpp_result_gen = Rcpp::wrap(test_sorted_vector_difference(larger, smaller));
    return rcpp_result_gen;
END_RCPP
}
// test_sorted_vector_difference2
std::vector<int> test_sorted_vector_difference2(const std::vector<int> larger, const std::vector<int> smaller);
RcppExport SEXP _gL0Learn_test_sorted_vector_difference2(SEXP largerSEXP, SEXP smallerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type larger(largerSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type smaller(smallerSEXP);
    rcpp_result_gen = Rcpp::wrap(test_sorted_vector_difference2(larger, smaller));
    return rcpp_result_gen;
END_RCPP
}
// test_insert_sorted_vector_into_sorted_vector
std::vector<int> test_insert_sorted_vector_into_sorted_vector(const std::vector<int> x1, const std::vector<int> x2);
RcppExport SEXP _gL0Learn_test_insert_sorted_vector_into_sorted_vector(SEXP x1SEXP, SEXP x2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type x2(x2SEXP);
    rcpp_result_gen = Rcpp::wrap(test_insert_sorted_vector_into_sorted_vector(x1, x2));
    return rcpp_result_gen;
END_RCPP
}
// test_coordinate_matrix_to_vector_to_matrix
arma::umat test_coordinate_matrix_to_vector_to_matrix(const arma::umat& coords_ma);
RcppExport SEXP _gL0Learn_test_coordinate_matrix_to_vector_to_matrix(SEXP coords_maSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat& >::type coords_ma(coords_maSEXP);
    rcpp_result_gen = Rcpp::wrap(test_coordinate_matrix_to_vector_to_matrix(coords_ma));
    return rcpp_result_gen;
END_RCPP
}
// test_unravel_indices
arma::umat test_unravel_indices(const arma::uvec& indices, const arma::uword p);
RcppExport SEXP _gL0Learn_test_unravel_indices(SEXP indicesSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type indices(indicesSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(test_unravel_indices(indices, p));
    return rcpp_result_gen;
END_RCPP
}
// is_sympd
bool is_sympd(const arma::mat& x);
RcppExport SEXP _gL0Learn_is_sympd(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(is_sympd(x));
    return rcpp_result_gen;
END_RCPP
}
// upper_triangluar_coords
arma::umat upper_triangluar_coords(const arma::uword p);
RcppExport SEXP _gL0Learn_upper_triangluar_coords(SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uword >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(upper_triangluar_coords(p));
    return rcpp_result_gen;
END_RCPP
}
// check_coordinate_matrix_is_valid
bool check_coordinate_matrix_is_valid(const arma::umat& coords_ma, const bool for_order, const bool for_upper_triangle);
RcppExport SEXP _gL0Learn_check_coordinate_matrix_is_valid(SEXP coords_maSEXP, SEXP for_orderSEXP, SEXP for_upper_triangleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat& >::type coords_ma(coords_maSEXP);
    Rcpp::traits::input_parameter< const bool >::type for_order(for_orderSEXP);
    Rcpp::traits::input_parameter< const bool >::type for_upper_triangle(for_upper_triangleSEXP);
    rcpp_result_gen = Rcpp::wrap(check_coordinate_matrix_is_valid(coords_ma, for_order, for_upper_triangle));
    return rcpp_result_gen;
END_RCPP
}
// check_is_valid_coordinate_subset
bool check_is_valid_coordinate_subset(const arma::umat& larger_coord_set, const arma::umat& smaller_coord_set);
RcppExport SEXP _gL0Learn_check_is_valid_coordinate_subset(SEXP larger_coord_setSEXP, SEXP smaller_coord_setSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat& >::type larger_coord_set(larger_coord_setSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type smaller_coord_set(smaller_coord_setSEXP);
    rcpp_result_gen = Rcpp::wrap(check_is_valid_coordinate_subset(larger_coord_set, smaller_coord_set));
    return rcpp_result_gen;
END_RCPP
}
// gL0Learn_fit_R
Rcpp::List gL0Learn_fit_R(const arma::mat& Y, const arma::mat& theta_init, const SEXP l0, const SEXP l1, const SEXP l2, const std::string algorithm, const SEXP lows, const SEXP highs, const arma::umat& initial_active_set, const arma::umat& super_active_set, const double atol, const double rtol, const size_t max_iter);
RcppExport SEXP _gL0Learn_gL0Learn_fit_R(SEXP YSEXP, SEXP theta_initSEXP, SEXP l0SEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP algorithmSEXP, SEXP lowsSEXP, SEXP highsSEXP, SEXP initial_active_setSEXP, SEXP super_active_setSEXP, SEXP atolSEXP, SEXP rtolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type l0(l0SEXP);
    Rcpp::traits::input_parameter< const SEXP >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< const SEXP >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< const std::string >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type lows(lowsSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type highs(highsSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type initial_active_set(initial_active_setSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type super_active_set(super_active_setSEXP);
    Rcpp::traits::input_parameter< const double >::type atol(atolSEXP);
    Rcpp::traits::input_parameter< const double >::type rtol(rtolSEXP);
    Rcpp::traits::input_parameter< const size_t >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(gL0Learn_fit_R(Y, theta_init, l0, l1, l2, algorithm, lows, highs, initial_active_set, super_active_set, atol, rtol, max_iter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gL0Learn_test_Oracle_prox", (DL_FUNC) &_gL0Learn_test_Oracle_prox, 6},
    {"_gL0Learn_test_oracle_L0", (DL_FUNC) &_gL0Learn_test_oracle_L0, 6},
    {"_gL0Learn_test_oracle_L0L2", (DL_FUNC) &_gL0Learn_test_oracle_L0L2, 6},
    {"_gL0Learn_test_oracle_L0L1L2", (DL_FUNC) &_gL0Learn_test_oracle_L0L1L2, 6},
    {"_gL0Learn_test_union_of_correlated_features", (DL_FUNC) &_gL0Learn_test_union_of_correlated_features, 2},
    {"_gL0Learn_test_union_of_correlated_features2", (DL_FUNC) &_gL0Learn_test_union_of_correlated_features2, 2},
    {"_gL0Learn_test_sorted_vector_difference", (DL_FUNC) &_gL0Learn_test_sorted_vector_difference, 2},
    {"_gL0Learn_test_sorted_vector_difference2", (DL_FUNC) &_gL0Learn_test_sorted_vector_difference2, 2},
    {"_gL0Learn_test_insert_sorted_vector_into_sorted_vector", (DL_FUNC) &_gL0Learn_test_insert_sorted_vector_into_sorted_vector, 2},
    {"_gL0Learn_test_coordinate_matrix_to_vector_to_matrix", (DL_FUNC) &_gL0Learn_test_coordinate_matrix_to_vector_to_matrix, 1},
    {"_gL0Learn_test_unravel_indices", (DL_FUNC) &_gL0Learn_test_unravel_indices, 2},
    {"_gL0Learn_is_sympd", (DL_FUNC) &_gL0Learn_is_sympd, 1},
    {"_gL0Learn_upper_triangluar_coords", (DL_FUNC) &_gL0Learn_upper_triangluar_coords, 1},
    {"_gL0Learn_check_coordinate_matrix_is_valid", (DL_FUNC) &_gL0Learn_check_coordinate_matrix_is_valid, 3},
    {"_gL0Learn_check_is_valid_coordinate_subset", (DL_FUNC) &_gL0Learn_check_is_valid_coordinate_subset, 2},
    {"_gL0Learn_gL0Learn_fit_R", (DL_FUNC) &_gL0Learn_gL0Learn_fit_R, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_gL0Learn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
