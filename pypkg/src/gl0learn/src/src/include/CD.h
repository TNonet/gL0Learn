#ifndef CD_H
#define CD_H
#include "active_set.h"
#include "arma_includes.h"
#include "fitmodel.h"
#include "gap.h"
#include "oracle.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <chrono>
#include <iomanip>
#include <thread>
#include <utility>

template <class O> struct CDParams {
  const double tol;
  const size_t max_active_set_size;
  const GapMethod gap_method;
  const bool one_normalize;
  const size_t max_iter;
  const O oracle;
  const std::string algorithm;
  const size_t max_swaps;
  const bool shuffle_feature_order;

  CDParams(const double tol, const size_t max_active_set_size,
           const GapMethod gap_method, const bool one_normalize,
           const size_t max_iter, const O &oracle, std::string algorithm,
           const size_t max_swaps, const bool shuffle_feature_order)
      : tol{tol}, max_active_set_size{max_active_set_size},
        gap_method{gap_method}, one_normalize{one_normalize},
        max_iter{max_iter}, oracle{oracle}, algorithm{std::move(algorithm)},
        max_swaps{max_swaps}, shuffle_feature_order{shuffle_feature_order} {};
};

template <class TY, class TR, class TT, class TP> class CD {
public:
  CD(const TY &Y, const TT &theta, const TP &params,
     const coordinate_vector &active_set,
     const coordinate_vector &super_active_set)
      : Y{Y}, S_diag{arma::sum(arma::square(Y), 0)}, params{params},
        super_active_set{super_active_set} {
    this->theta = TT(theta);                          // Take a copy of theta!
    this->active_set = coordinate_vector(active_set); // Ensure a copy is taken.
    this->R = this->Y * this->theta;
    this->costs.reserve(this->params.max_iter);
    this->active_set_size.reserve(this->params.max_iter);
  };

  void restrict_active_set();
  std::tuple<coordinate_vector, std::vector<double>>
  active_set_expansion(const coordinate_vector &search_space) const;
  void inner_fit();
  fitmodel fit();
  std::tuple<bool, arma::uword> psi_row_fit(arma::uword row_ix);
  fitmodel fitpsi();
  double inline compute_objective() const;
  bool inline converged(double old_objective, double cur_objective,
                        size_t cur_iter) const;

  double inline a(arma::uword i, arma::uword j) const;
  arma::vec inline a(arma::uword i, const arma::uvec &js) const;
  double inline b(arma::uword i, arma::uword j) const;
  arma::vec inline b(arma::uword i, const arma::uvec &js) const;
  double inline b_update(arma::uword i, arma::uword j) const;

  void set_theta_zero(arma::uword i, arma::uword j);
  void set_theta(arma::uword, arma::uword j, double theta_ij);
  void update_diag(arma::uword i);

private:
  const TY Y;
  TR R;
  TT theta;
  const arma::rowvec S_diag;
  const TP params;
  coordinate_vector active_set;
  coordinate_vector super_active_set;
  std::vector<double> costs; // TODO: Should be a map of where this value was
  // from. {"initial_objective": X, "CD_1"
  std::vector<std::size_t> active_set_size;
};

template <class TY, class TR, class TT, class TP>
fitmodel CD<TY, TR, TT, TP>::fit() {
  // Accepts current state for active_set and theta.
  /*
   *  CD Algorithm:
   *  Given:
   *      active_set, AS,: vector of coordinates sorted lexicographically
   *      super_active_set, SAS: vector of coordinates sorted lexicographically
   *          such that every item in the active active_set exists in
   * super_active_set. theta: (p, p) symmetric matrix with non-zero diagonals
   *          such that an item theta[i, j] not in SAS or in SAS but not in AS
   * must be zero.
   *
   *  Steps:
   *      old_objective <- Inf;
   *      cur_objective <- Inf:
   *
   *      For iter in 0, ..., max_iter-1:
   *          theta[AS] <- updated by Hussein's operator;
   *          old_objective, cur_objective <- compute_objective(), old_objective
   *
   *          if converged(old_objective, cur_objective):
   *              incr_AS <- items in SAS - AS that want to to be non-zero;
   *              if incr_AS is empty:
   *                  AS <- support of theta
   *                  DONE (RETURN AS, theta)
   *              else:
   *                  AS <- AS U incr_AS
   */
  double old_objective;
  double cur_objective = std::numeric_limits<double>::infinity();

  std::size_t cur_iter = 0;
  // RUN CD on AS until convergence
  while (cur_iter <= this->params.max_iter) {
    UserInterrupt();
    COUT << "inner_fit()\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    this->inner_fit(); // Fits on active_set
    old_objective = cur_objective;
    cur_objective = this->compute_objective();
    this->costs.push_back(cur_objective);
    this->active_set_size.push_back(this->active_set.size());
    cur_iter++;

    if (this->converged(old_objective, cur_objective, cur_iter)) {
      COUT << "values_to_check\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      const coordinate_vector values_to_check =
          sorted_vector_difference2(this->super_active_set, this->active_set);
      COUT << "values_to_check_finished\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (values_to_check.empty()) {
        break;
      }

      COUT << "active_set_expansion\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      const std::tuple<coordinate_vector, std::vector<double>> tmp =
          this->active_set_expansion(values_to_check);
      coordinate_vector add_to_active_set = std::get<0>(tmp);

      const std::size_t new_active_set_size =
          this->active_set.size() + add_to_active_set.size();

      if (new_active_set_size > this->params.max_active_set_size) {
        const std::vector<double> q_values = std::get<1>(tmp);
        const std::size_t n_to_keep =
            this->params.max_active_set_size - this->active_set.size();

        COUT << "nth_largest_indices\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const std::vector<size_t> indices =
            nth_largest_indices(q_values, n_to_keep);

        coordinate_vector n_add_to_active_set;
        n_add_to_active_set.reserve(n_to_keep);

        auto it = indices.begin();
        const auto it_end = indices.end();
        for (; it != it_end; ++it) {
          n_add_to_active_set.push_back(add_to_active_set[*it]);
        }
        add_to_active_set = n_add_to_active_set;
      }

      COUT << "insert_sorted_vector_into_sorted_vector\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      this->active_set = insert_sorted_vector_into_sorted_vector(
          this->active_set, add_to_active_set);
    }
  }
  COUT << "inner_fit loop finished\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  this->restrict_active_set();
  cur_objective = this->compute_objective();
  this->costs.push_back(cur_objective);
  this->active_set_size.push_back(this->active_set.size());
  COUT << "fitmodel created\n";
  return fitmodel(this->theta, this->R, this->costs, this->active_set_size);
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::restrict_active_set() {
  coordinate_vector restricted_active_set;
  // copy only coordinates with non_zero thetas:
  std::copy_if(this->active_set.begin(), this->active_set.end(),
               std::back_inserter(restricted_active_set),
               [this](const coordinate ij) {
                 return (std::get<0>(ij) == std::get<1>(ij)) ||
                        (this->theta(std::get<0>(ij), std::get<1>(ij)) != 0);
               });
  this->active_set = restricted_active_set;
}

template <class TY, class TR, class TT, class TP>
std::tuple<coordinate_vector, std::vector<double>>
CD<TY, TR, TT, TP>::active_set_expansion(
    const coordinate_vector &search_space) const {

  const size_t p = this->Y.n_cols;

  coordinate_vector items_to_expand_active_set_by;
  std::vector<double> items_Q;

  const std::size_t reserve = p;
  //      std::max(1UL, static_cast<std::size_t>(search_space.size() / p));
  items_Q.reserve(reserve); // What should we predict as the number of items to
  // reserve here?
  items_to_expand_active_set_by.reserve(reserve);

  const arma::uvec feature_order = coordinate_iter_order(
      search_space.size(), this->params.shuffle_feature_order);

  arma::uvec::const_iterator it = feature_order.begin();
  const arma::uvec::const_iterator it_end = feature_order.end();

  for (; it != it_end; ++it) {
    const auto ij = search_space[*it];
    const arma::uword i = std::get<0>(ij);
    const arma::uword j = std::get<1>(ij);

    const auto a = this->a(i, j);
    const auto b = this->b(i, j);

    const double q_ij = this->params.oracle.Q(a, b, i, j);

    if (q_ij != 0) {
      items_Q.push_back(std::abs(q_ij));
      items_to_expand_active_set_by.push_back(ij);
    }
  }
  return std::make_tuple(items_to_expand_active_set_by, items_Q);
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::inner_fit() {
  const size_t p = this->Y.n_cols;

  const arma::uvec feature_order = coordinate_iter_order(
      active_set.size(), this->params.shuffle_feature_order);

  arma::uvec::const_iterator it = feature_order.begin();
  const arma::uvec::const_iterator it_end = feature_order.end();

  for (; it != it_end; ++it) {
    auto const ij = this->active_set[*it];
    const arma::uword i = std::get<0>(ij);
    const arma::uword j = std::get<1>(ij);

    const double old_theta_ij = this->theta(i, j);
    // old_theta_ij is identical to old_theta_ji;
    // const double old_theta_ji = this->theta(j, i);

    const auto a = this->a(i, j);
    const auto b = this->b_update(i, j);

    const double new_theta = this->params.oracle.Q(a, b, i, j);

    this->theta(i, j) = new_theta;
    this->theta(j, i) = new_theta;

    this->R.col(i) += (new_theta - old_theta_ij) * this->Y.col(j);
    this->R.col(j) += (new_theta - old_theta_ij) * this->Y.col(i);
  }

  for (auto i = 0; i < p; i++) {
    // Usually at least 1 item per column, so we always update every diagonal
    this->update_diag(i);
  }
}

template <class TY, class TR, class TT, class TP>
fitmodel CD<TY, TR, TT, TP>::fitpsi() {
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  static_cast<void>(this->fit());
  const arma::uword p = this->Y.n_cols;

  for (auto i = 0; i < this->params.max_swaps; i++) {
    UserInterrupt();

    arma::uvec feature_order;
    // No need to swap on the last row
    if (this->params.shuffle_feature_order) {
      feature_order = arma::randperm(p - 1);
    } else {
      feature_order = arma::linspace<arma::uvec>(0, p - 2, p - 1);
    }

    arma::uvec::const_iterator it = feature_order.begin();
    const arma::uvec::const_iterator it_end = feature_order.end();

    bool swap = false;

    for (; it != it_end; ++it) {
      const arma::uword row_ix = *it;

      // TODO: This needs to properly update the active set.
      const auto row_fit = this->psi_row_fit(row_ix);
      swap = swap || std::get<0>(row_fit);
      if (swap) {
        for (arma::uword row : {row_ix, std::get<1>(row_fit)}) {
          // Only updates the two items that changed!
          this->update_diag(row);
        }
        break;
      }
    }

    if (!swap) {
      break;
    } else {
      static_cast<void>(this->fit());
    }
  }

  return fitmodel(this->theta, this->R, this->costs, this->active_set_size);
}

template <class TY, class TR, class TT, class TP>
std::tuple<bool, arma::uword>
CD<TY, TR, TT, TP>::psi_row_fit(const arma::uword row_ix) {
  // TODO: Check if this can be {row_ix, row_ix+1}
  // TODO: Check if we should be storing active_set in column major sort order?
  const coordinate row_ix_0 = {row_ix, 0};
  auto it = std::lower_bound(this->super_active_set.begin(),
                             this->super_active_set.end(), row_ix_0);

  // TODO: Check if this can be {row_ix+1, row_ix+2}
  // TODO: Check if we should be storing active_set in column major sort order?
  const coordinate row_ix_p1_0 = {row_ix + 1, 0};
  auto it_tmp = it;
  auto end =
      std::lower_bound(it_tmp, this->super_active_set.end(), row_ix_p1_0);

  std::vector<arma::uword> zero_indices;
  std::vector<arma::uword> non_zero_indices;

  for (; it != end; ++it) {
    // zero_indices and non_zero_indices will not form the set (1, ..., p)
    // as if theta(i, i) is non-zero it will not go into non_zero_indicies
    const auto j = std::get<1>(*it);
    if (this->theta(row_ix, j) != 0) {
      non_zero_indices.push_back(j);
    } else {
      zero_indices.push_back(j);
    }
  }
  // If every item is either 0 or non-zero then swapping is pointless
  if (zero_indices.empty() || non_zero_indices.empty()) {
    return {false, -1L};
  }

  const arma::uvec zeros(zero_indices);
  const arma::uvec non_zeros(non_zero_indices);
  const arma::vec theta_diag = arma::vec(this->theta.diag());

  arma::uvec non_zero_order;
  if (this->params.shuffle_feature_order) {
    non_zero_order = arma::randperm(non_zero_indices.size());
  } else {
    // We know that non_zero_indices has at least 1 element in it!
    non_zero_order = arma::linspace<arma::uvec>(0, non_zero_indices.size() - 1,
                                                non_zero_indices.size());
  }

  arma::uvec::const_iterator non_zeros_it = non_zero_order.begin();
  const arma::uvec::const_iterator it_end = non_zero_order.end();

  for (; non_zeros_it != it_end; ++non_zeros_it) {
    const arma::uword j = non_zero_indices[*non_zeros_it];
    this->set_theta_zero(row_ix, j);

    const auto a = this->a(row_ix, j);
    const auto b = this->b(row_ix, j);

    const std::tuple<double, double> theta_f =
        this->params.oracle.Qobj(a, b, row_ix, j);

    const double theta_j = std::get<0>(theta_f);
    const double f = std::get<1>(theta_f);

    const arma::vec a_vec = this->a(row_ix, zeros);
    const arma::vec b_vec = this->b(row_ix, zeros);

    const std::tuple<arma::vec, arma::vec> thetas_fs =
        this->params.oracle.Qobj(a_vec, b_vec, row_ix, zeros);

    const arma::vec thetas = std::get<0>(thetas_fs);
    const arma::vec fs = std::get<1>(thetas_fs);

    if (f < fs.min()) {
      this->set_theta(row_ix, j, theta_j);
    } else {
      /* If a swap is accepted, (`k` <- NNZ for `j` <- 0) in row `row_ix`,
       * We will need to updated AS to ensure it contains `k`. Lucikly,
       * We can keep `j` in the AS.
       * Since AS is a sorted a coordinate vector such that for each pair
       * of elements (p1, p2) in AS p1 will be found before p2 if and only
       * if p1 < p2. That is a lexicographical comparison between the 2D
       * coordinates p1 and p2.
       * Therefore, we search AS for the first item, u, that is larger than
       * (row_ix, k), using `lower_bound`. We then insert (row_ix, k)
       *  before u.
       */

      const auto ell = arma::index_min(fs);
      const auto k = zeros(ell);
      const auto k_theta = thetas(ell);

      this->set_theta(row_ix, k, k_theta);

      // Find location to remove (row_ix, j) from active set!
      const coordinate row_ix_j = {row_ix, j};
      const auto low_j = std::lower_bound(this->active_set.begin(),
                                          this->active_set.end(), row_ix_j);

      /* Since we just found the location of `(row_ix, j) and we know that item
       * `(row_ix, k)` is quite close to (row_ix, j). We can use this
       * information to speed up our search! If j > k. Then we know `(row_ix,
       * k)` appears before (row_ix, j) and thus we can limit our search to:
       *      [begin, loc)
       * If k > j. Then we know `(row_ix, k)` appears after `(row_ix, j) and
       * thus we can limit our search to: (loc, end]
       */
      coordinate_vector::iterator k_begin;
      coordinate_vector::iterator k_end;
      if (j > k) {
        k_begin = this->active_set.begin();
        k_end = this->active_set.erase(low_j);
      } else {
        k_begin = this->active_set.erase(low_j);
        k_end = this->active_set.end();
      }

      // Find location to insert (row_ix, k)
      const coordinate row_ix_k = {row_ix, k};
      const auto low_k = std::lower_bound(k_begin, k_end, row_ix_k);
      this->active_set.insert(low_k, row_ix_k);
      return {true, k};
    }
  }

  return {false, -1L};
}

template <class TY, class TR, class TT, class TP>
bool inline CD<TY, TR, TT, TP>::converged(const double old_objective,
                                          const double cur_objective,
                                          const size_t cur_iter) const {
  return ((cur_iter > 1) &&
          (relative_gap(old_objective, cur_objective, this->params.gap_method,
                        this->params.one_normalize) <= this->params.tol));
}

template <class TY, class TR, class TT, class TP>
double inline CD<TY, TR, TT, TP>::compute_objective() const {
  /*
   *  Objective = \sum_{i=1}^{p}(||<Y, theta[i, :]>||_2 - log(theta[i, i]))
   *
   *  Notes
   *  -----
   *  If we use a sparse form of TT, the objective can be sped up in the active
   * set calculation.
   */
  return objective(this->theta, this->R, this->active_set,
                   this->params.oracle.penalty);
}

template <class TY, class TR, class TT, class TP>
double inline CD<TY, TR, TT, TP>::a(const arma::uword i,
                                    const arma::uword j) const {
  return this->S_diag[j] / this->theta(i, i) +
         this->S_diag[i] / this->theta(j, j);
}

template <class TY, class TR, class TT, class TP>
arma::vec inline CD<TY, TR, TT, TP>::a(const arma::uword i,
                                       const arma::uvec &js) const {
  const arma::vec theta_diag = this->theta.diag();
  return (this->S_diag(i) / theta_diag(js) + this->S_diag(js) / theta_diag(i));
}

template <class TY, class TR, class TT, class TP>
double inline CD<TY, TR, TT, TP>::b(const arma::uword i,
                                    const arma::uword j) const {
  return 2 * ((arma::dot(this->Y.col(j), this->R.col(i)) / this->theta(i, i)) +
              (arma::dot(this->Y.col(i), this->R.col(j)) / this->theta(j, j)));
}

template <class TY, class TR, class TT, class TP>
arma::vec inline CD<TY, TR, TT, TP>::b(const arma::uword i,
                                       const arma::uvec &js) const {
  const arma::vec theta_diag = this->theta.diag();
  return 2 * (((this->Y.cols(js).t() * this->R.col(i)) / theta_diag(i)) +
              ((this->R.cols(js).t() * this->Y.col(i)) / theta_diag(js)));
}

template <class TY, class TR, class TT, class TP>
double CD<TY, TR, TT, TP>::b_update(const arma::uword i,
                                    const arma::uword j) const {
  const double theta_ij = this->theta(i, j);
  return 2 * ((arma::dot(this->Y.col(j), this->R.col(i)) -
               theta_ij * this->S_diag(j)) /
                  this->theta(i, i) +
              (arma::dot(this->Y.col(i), this->R.col(j)) -
               theta_ij * this->S_diag(i)) /
                  this->theta(j, j));
}

template <class TY, class TR, class TT, class TP>
void inline CD<TY, TR, TT, TP>::set_theta_zero(const arma::uword i,
                                               const arma::uword j) {
  R.col(i) -= this->theta(i, j) * this->Y.unsafe_col(j);
  R.col(j) -= this->theta(j, i) * this->Y.unsafe_col(i);
  this->theta(i, j) = 0;
  this->theta(j, i) = 0;
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::set_theta(const arma::uword i, const arma::uword j,
                                   const double theta_ij) {
  R.col(i) += theta_ij * this->Y.unsafe_col(j);
  R.col(j) += theta_ij * this->Y.unsafe_col(i);
  this->theta(i, j) = theta_ij;
  this->theta(j, i) = theta_ij;
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::update_diag(const arma::uword i) {
  this->R.col(i) -= this->theta(i, i) * this->Y.col(i);
  this->theta(i, i) =
      R_nl(this->S_diag(i), arma::dot(this->R.col(i), this->R.col(i)));
  this->R.col(i) += this->theta(i, i) * this->Y.col(i);
}

#endif // CD_H