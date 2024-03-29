# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

test_union_of_correlated_features <- function(x, threshold) {
    .Call(`_gL0Learn_test_union_of_correlated_features`, x, threshold)
}

test_union_of_correlated_features2 <- function(x, threshold) {
    .Call(`_gL0Learn_test_union_of_correlated_features2`, x, threshold)
}

test_coordinate_matrix_to_vector_to_matrix <- function(coords_ma) {
    .Call(`_gL0Learn_test_coordinate_matrix_to_vector_to_matrix`, coords_ma)
}

test_unravel_indices <- function(indices, p) {
    .Call(`_gL0Learn_test_unravel_indices`, indices, p)
}

is_sympd <- function(x) {
    .Call(`_gL0Learn_is_sympd`, x)
}

upper_triangluar_coords <- function(p) {
    .Call(`_gL0Learn_upper_triangluar_coords`, p)
}

check_coordinate_matrix_is_valid <- function(coords_ma, for_order = TRUE, for_upper_triangle = TRUE) {
    .Call(`_gL0Learn_check_coordinate_matrix_is_valid`, coords_ma, for_order, for_upper_triangle)
}

check_is_valid_coordinate_subset <- function(larger_coord_set, smaller_coord_set) {
    .Call(`_gL0Learn_check_is_valid_coordinate_subset`, larger_coord_set, smaller_coord_set)
}
