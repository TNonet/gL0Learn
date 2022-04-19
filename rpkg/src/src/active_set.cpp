#include "active_set.h"

arma::uvec coordinate_iter_order(const std::size_t num_coords, bool shuffle) {
  if (num_coords == 0) {
    return {};
  }
  if (shuffle) {
    return arma::randperm(num_coords);
  } else {
    return arma::linspace<arma::uvec>(0, num_coords - 1, num_coords);
  }
}

coordinate_vector upper_triangle_coordinate_vector(const arma::uword p) {
  coordinate_vector coord_vec;
  coord_vec.reserve(p * (p - 1));
  for (arma::uword i = 0; i < p; i++) {
    for (arma::uword j = i + 1; j < p; j++) {
      coord_vec.push_back({i, j});
    }
  }

  return coord_vec;
}

coordinate_vector
union_of_correlated_features(const arma::mat &x,
                             const double correlation_threshold) {
  /*
   *  Returns the coordinates of the upper triangle of xtx where xtx[i,j] >
   * correlation_threshold
   *
   */
  const arma::vec s_diag = arma::sum(arma::square(x), 0);
  const auto p = x.n_cols;
  coordinate_vector active_set = coordinate_vector();
  active_set.reserve(p * (p - 1));

  for (auto i = 0; i < p - 1; ++i) {
    const arma::vec xxt_i = x.cols(i + 1, p - 1).t() * x.col(i) / s_diag[i];
    const arma::uvec highly_correlated_xxt_i_indicies =
        arma::find(arma::abs(xxt_i) > correlation_threshold);

    arma::uvec::const_iterator it = highly_correlated_xxt_i_indicies.begin();
    const arma::uvec::const_iterator it_end =
        highly_correlated_xxt_i_indicies.end();

    for (; it != it_end; ++it) {
      active_set.push_back({i, (*it) + i + 1});
    }
  }

  return active_set;
}

arma::umat unravel_ut_indices(const arma::uvec &transposed_indices,
                              const arma::uword p) {
  /*
   * Unravels upper triangular indices of an matrix into lexigraphically
   *  sorted coordinates of a matrix.
   *
   * Requies `transposed_indices` be the indices of the of the transposed
   *  matrix. For example, instead of;
   *      indices <- find(x > 1)
   *  It shoudl be:
   *      indices <- find(x.t() > 1).
   *
   * Given the indices of a square p by p matrix:
   *   +-----+--------+-----+----------+
   *   | 0 X | p      | ... | p(p-1)   |
   *   +-----+--------+-----+----------+
   *   | -   | p+1  X | ... | p(p-1)+1 |
   *   +-----+--------+-----+----------+
   *   | ... | ...    | ... | ...      |
   *   +-----+--------+-----+----------+
   *   | --  | ---    | ... | p*p-1  X |
   *   +-----+--------+-----+----------+
   *
   *   To convert to coordinates (i, j) we notice that:
   *      i [row] = index % p
   *      j [col] = index // p (floor divide)
   *
   *  Note, if you transpose a matrix using .t(), the indices flip!
   *  +--------+----------+-----+--------+
   *  | 0      | 1        | ... | p-1    |
   *  +--------+----------+-----+--------+
   *  | p      | p+1      | ... | 2(p-1) |
   *  +--------+----------+-----+--------+
   *  | ...    | ...      | ... | ...    |
   *  +--------+----------+-----+--------+
   *  | p(p-1) | p(p-1)+1 | ... | p*p-1  |
   *  +--------+----------+-----+--------+
   *
   */
  arma::umat coords(transposed_indices.n_elem, 2);

  coords.col(0) = arma::floor(transposed_indices / p);
  coords.col(1) = transposed_indices - coords.col(0) * p;

  return coords;
}

bool check_is_coordinate_subset(const arma::umat &larger_coord_set,
                                const arma::umat &smaller_coord_set) {
  /*
   *  Determines if `smaller_coord_set` is contained in `larger_coord_set`.
   *
   *  Both `*_coord_set` must be sorted coordinate_matrices. For example,
   *  from:
   *   `union_of_correlated_features2`
   *   `coordinate_matrix_from_vector`
   */

  if (smaller_coord_set.is_empty()) {
    return true;
  }
  if (larger_coord_set.is_empty()) {
    STOP("expect larger coordinate set to be non-empty.");
  }

  const auto max_col =
      std::max(larger_coord_set.col(1).max(), smaller_coord_set.col(1).max()) +
      1;

  const arma::uvec larger_indices =
      larger_coord_set.col(0) * max_col + larger_coord_set.col(1);
  const arma::uvec smaller_indices =
      smaller_coord_set.col(0) * max_col + smaller_coord_set.col(1);

  return std::includes(larger_indices.begin(), larger_indices.end(),
                       smaller_indices.begin(), smaller_indices.end());
}

bool check_coordinate_matrix(const arma::umat &coords_ma, const bool for_order,
                             const bool for_upper_triangle) {
  /*
   *  Checks `coords_ma` to ensure:
   *      if `for_order` is true, that the ordering is lexicographically
   *       correct. This also checks for duplicates (which are not allowed)
   *      if `for_upper_triangle` is true, that the values are limited to the
   *       upper triangle of a matrix.
   */

  if (coords_ma.is_empty()) {
    return true;
  }

  bool check = true;

  if (for_order) {
    const auto max_col = coords_ma.col(1).max() + 1;
    const arma::uvec indices = coords_ma.col(0) * max_col + coords_ma.col(1);
    check = check & indices.is_sorted("strictascend");
  }

  if (for_upper_triangle) {
    check = check & arma::all(coords_ma.col(0) < coords_ma.col(1));
  }

  return check;
}

arma::umat union_of_correlated_features2(const arma::mat &x,
                                         const double correlation_threshold) {
  /*
   *  Returns the coordinates of the upper triangle, not including diagonal
   *   of xtx where xtx[i,j] > correlation_threshold
   *
   *  Note, because we are using armadillo logic which is column major,
   *   we have to flip our coordinates to make this work.
   */
  const arma::rowvec s_diag = arma::sum(arma::square(x), 0);
  const auto p = x.n_cols;
  arma::mat upper_triangle_indicator(p, p, arma::fill::zeros);
  upper_triangle_indicator
      .elem(arma::trimatu_ind(arma::size(upper_triangle_indicator), 1))
      .fill(1);
  const arma::mat xtx_upper_triangle =
      (arma::abs(x.t() * x) % upper_triangle_indicator).t();
  const arma::uvec highly_correlated_indicies = arma::find(
      xtx_upper_triangle.each_row() / s_diag > correlation_threshold);
  return unravel_ut_indices(highly_correlated_indicies, p);
}

coordinate_vector coordinate_vector_from_matrix(const arma::umat &coords_ma) {
  const arma::uword n = coords_ma.n_rows;

  coordinate_vector coords_vec;
  coords_vec.reserve(n);

  for (arma::uword row_index = 0; row_index < n; ++row_index) {
    arma::umat::const_row_iterator it_row_begin =
        coords_ma.begin_row(row_index);
    const auto i = (*it_row_begin);
    ++it_row_begin;
    const auto j = (*it_row_begin);
    coords_vec.push_back({i, j});
  }
  return coords_vec;
}

arma::umat coordinate_matrix_from_vector(const coordinate_vector &coord_vec) {
  const auto n = coord_vec.size();
  arma::umat coords_ma(n, 2);

  auto it = coord_vec.begin();

  for (arma::uword row_index = 0; row_index < n; ++row_index, ++it) {
    arma::umat::row_iterator it_row_begin = coords_ma.begin_row(row_index);
    (*it_row_begin) = std::get<0>(*it);
    ++it_row_begin;
    (*it_row_begin) = std::get<1>(*it);
  }

  return coords_ma;
}

coordinate_vector support_from_active_set(const arma::mat &x,
                                          const coordinate_vector &active_set) {

  coordinate_vector support = coordinate_vector(active_set.size());
  for (auto ij : active_set) {
    const auto i = std::get<0>(ij);
    const auto j = std::get<1>(ij);

    if (x(i, j) != 0) {
      support.push_back(ij);
    }
  }
  support.shrink_to_fit();
  return support;
}

coordinate_vector sorted_vector_difference2(const coordinate_vector &larger,
                                            const coordinate_vector &smaller) {
  /*
   *  Returns all items in `larger` that aren't in `smaller`.
   *  Assumes both larger and smaller are sorted in the same fashion.
   */
  auto larger_size = larger.size();
  auto smaller_size = smaller.size();
  coordinate_vector difference = {};
  difference.reserve(larger_size - smaller_size);

  std::set_difference(larger.begin(), larger.end(), smaller.begin(),
                      smaller.end(),
                      std::inserter(difference, difference.begin()));

  return difference;
}

coordinate_vector
insert_sorted_vector_into_sorted_vector(const coordinate_vector &x1,
                                        const coordinate_vector &x2) {
  /*
   *  Combines `base` and `by`
   *  Assumes both larger and smaller are sorted in the same fashion.
   */
  coordinate_vector merged;
  merged.reserve(x1.size() + x2.size());
  std::merge(x1.begin(), x1.end(), x2.begin(), x2.end(),
             std::back_inserter(merged));

  return merged;
}
