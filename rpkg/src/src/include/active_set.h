#ifndef H_ACTIVE_SET
#define H_ACTIVE_SET
#include "arma_includes.h"
#include <tuple>
#include <vector>

typedef std::tuple<arma::uword, arma::uword> coordinate;
typedef std::vector<coordinate> coordinate_vector;

arma::uvec coordinate_iter_order(std::size_t num_coords, bool shuffle);

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::tuple<T, T> &c) {
  return stream << "{" << std::get<0>(c) << ", " << std::get<1>(c) << "}";
}

template <typename T>
std::ostream &operator<<(std::ostream &stream,
                         const std::vector<std::tuple<T, T>> &c) {
  for (auto c_i : c) {
    stream << c_i << "\n";
  }

  return stream;
}

coordinate_vector upper_triangle_coordinate_vector(arma::uword p);

coordinate_vector union_of_correlated_features(const arma::mat &x,
                                               double correlation_threshold);

arma::umat unravel_ut_indices(const arma::uvec &transposed_indices,
                              arma::uword p);

bool check_is_coordinate_subset(const arma::umat &larger_coord_set,
                                const arma::umat &smaller_coord_set);

bool check_coordinate_matrix(const arma::umat &coords_ma, bool for_order = true,
                             bool for_upper_triangle = true);

arma::umat union_of_correlated_features2(const arma::mat &x,
                                         double correlation_threshold);

coordinate_vector coordinate_vector_from_matrix(const arma::umat &coords_ma);

arma::umat coordinate_matrix_from_vector(const coordinate_vector &coord_vec);

coordinate_vector support_from_active_set(const arma::mat &x,
                                          const coordinate_vector &active_set);

coordinate_vector sorted_vector_difference2(const coordinate_vector &larger,
                                            const coordinate_vector &smaller);

coordinate_vector
insert_sorted_vector_into_sorted_vector(const coordinate_vector &x1,
                                        const coordinate_vector &x2);

#endif // H_ACTIVE_SET