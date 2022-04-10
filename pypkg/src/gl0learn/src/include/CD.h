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
    this->theta = TT(theta);
    this->active_set = coordinate_vector(active_set); // Ensure a copy is taken.
    this->R = this->Y * this->theta;
    this->costs.reserve(this->params.max_iter);
    this->active_set_size.reserve(this->params.max_iter);
  };

  void restrict_active_set();
  std::tuple<coordinate_vector, std::vector<double>>
  active_set_expansion(const coordinate_vector &search_space);
  void inner_fit();
  fitmodel fit();
  std::tuple<bool, arma::uword> psi_row_fit(arma::uword row_ix);
  fitmodel fitpsi();
  double inline compute_objective();
  bool inline converged(double old_objective, double cur_objective,
                        size_t cur_iter);

  std::tuple<double, double> calc_ab(std::tuple<arma::uword, arma::uword> ij,
                                     const arma::vec &theta_diag);

  std::tuple<arma::vec, arma::vec> calc_ab(arma::uword i, arma::uvec js,
                                           const arma::vec &theta_diag);

  // Use SFINAE to overload when new_theta == 0;
  void remove_from_support(std::tuple<arma::uword, arma::uword> ij);
  void update_support(std::tuple<arma::uword, arma::uword> ij, double theta);
  void add_to_support(std::tuple<arma::uword, arma::uword> ij, double theta);

  // Helper functions!
  // active_set_of_row;
  // active_set_row_begin_iter;
  // active_set_row_end_iter;
  // a, b from {i, j}
  // theta_diag
  // update Residuals

private:
  const TY Y;
  TR R;
  TT theta;
  const arma::rowvec S_diag;
  const TP params;
  coordinate_vector active_set;
  coordinate_vector super_active_set;
  std::vector<double> costs;
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
  COUT << "fit 1\n";
  // RUN CD on AS until convergence
  while (cur_iter <= this->params.max_iter) {
    COUT << "fit loop" << cur_iter << "\n";
    this->inner_fit(); // Fits on active_set
    old_objective = cur_objective;
    cur_objective = this->compute_objective();
    this->costs.push_back(cur_objective);
    this->active_set_size.push_back(this->active_set.size());
    cur_iter++;
    COUT << "current_iter: " << cur_iter << " cur_objective = " << cur_objective
         << "\n";

    if (this->converged(old_objective, cur_objective, cur_iter)) {
      const coordinate_vector values_to_check =
          sorted_vector_difference2(this->super_active_set, this->active_set);

      if (values_to_check.empty()) {
        break;
      }

      const std::tuple<coordinate_vector, std::vector<double>> tmp =
          this->active_set_expansion(values_to_check);
      coordinate_vector add_to_active_set = std::get<0>(tmp);

      const std::size_t new_active_set_size =
          this->active_set.size() + add_to_active_set.size();

      if (new_active_set_size > this->params.max_active_set_size) {
        const std::vector<double> q_values = std::get<1>(tmp);
        COUT << "this->params.max_active_set_size = "
             << this->params.max_active_set_size << "\n";
        COUT << "this->active_set.size() = " << this->active_set.size() << "\n";
        const std::size_t n_to_keep =
            this->params.max_active_set_size - this->active_set.size();
        COUT << "n_to_keep = " << n_to_keep << "\n";
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

      if (add_to_active_set.empty()) {
        break;
      }

      this->active_set = insert_sorted_vector_into_sorted_vector(
          this->active_set, add_to_active_set);
    }
  }
  this->restrict_active_set();
  cur_objective = this->compute_objective();
  this->costs.push_back(cur_objective);
  this->active_set_size.push_back(this->active_set.size());
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
    const coordinate_vector &search_space) {

  const size_t p = this->Y.n_cols;
  const arma::vec theta_diag = arma::vec(this->theta.diag());

  const arma::mat ytr = this->Y.t() * this->R;

  coordinate_vector items_to_expand_active_set_by;
  items_to_expand_active_set_by.reserve(p);

  std::vector<double> items_Q;
  items_Q.reserve(p);

  arma::uvec feature_order;
  if (this->params.shuffle_feature_order) {
    feature_order = arma::randperm(search_space.size());
  } else {
    feature_order = arma::linspace<arma::uvec>(0, search_space.size() - 1,
                                               search_space.size());
  }

  arma::uvec::const_iterator it = feature_order.begin();
  const arma::uvec::const_iterator it_end = feature_order.end();

  for (; it != it_end; ++it) {
    const auto ij = search_space[*it];
    const arma::uword i = std::get<0>(ij);
    const arma::uword j = std::get<1>(ij);
    const double a =
        this->S_diag(j) / theta_diag(i) + this->S_diag(i) / theta_diag(j);
    const double b =
        2 * ytr(j, i) / theta_diag(i) + 2 * ytr(i, j) / theta_diag(j);

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

  arma::uvec feature_order;
  const arma::uword starting_active_set_size = this->active_set.size();
  if (this->params.shuffle_feature_order) {
    feature_order = arma::randperm(starting_active_set_size);
  } else {
    if (starting_active_set_size == 0) {
      feature_order = arma::linspace<arma::uvec>(0, 0, 0);
    } else {
      feature_order = arma::linspace<arma::uvec>(
          0, starting_active_set_size - 1, starting_active_set_size);
    }
  }

  arma::uvec::const_iterator it = feature_order.begin();
  const arma::uvec::const_iterator it_end = feature_order.end();

  for (; it != it_end; ++it) {
    auto const ij = this->active_set[*it];
    const arma::uword i = std::get<0>(ij);
    const arma::uword j = std::get<1>(ij);

    const double old_theta_ij = this->theta(i, j);
    const double old_theta_ji = this->theta(j, i);
    const double old_theta_ii = this->theta(i, i);
    const double old_theta_jj = this->theta(j, j);

    const double a =
        this->S_diag(j) / old_theta_ii + this->S_diag(i) / old_theta_jj;
    const double b = 2 * ((arma::dot(this->Y.col(j), this->R.col(i)) -
                           old_theta_ji * this->S_diag(j)) /
                              old_theta_ii +
                          (arma::dot(this->Y.col(i), this->R.col(j)) -
                           old_theta_ij * this->S_diag(i)) /
                              old_theta_jj);
    const double new_theta = this->params.oracle.Q(a, b, i, j);

    this->theta(i, j) = new_theta;
    this->theta(j, i) = new_theta;

    this->R.col(i) += (new_theta - old_theta_ij) * this->Y.col(j);
    this->R.col(j) += (new_theta - old_theta_ji) * this->Y.col(i);
  }

  for (auto i = 0; i < p; i++) {
    this->R.col(i) -= this->theta(i, i) * this->Y.col(i);
    this->theta(i, i) =
        R_nl(this->S_diag(i), arma::dot(this->R.col(i), this->R.col(i)));
    this->R.col(i) += this->theta(i, i) * this->Y.col(i);
  }
}

template <class TY, class TR, class TT, class TP>
fitmodel CD<TY, TR, TT, TP>::fitpsi() {
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  COUT << "fitpsi called \n";
  static_cast<void>(this->fit());
  COUT << "Pre psi cost: " << this->compute_objective() << " \n";
  const arma::uword p = this->Y.n_cols;

  for (auto i = 0; i < this->params.max_swaps;
       i++) { // TODO: Elevant 100 to Parameter
    COUT << "PSI iter: " << i << " \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));

    arma::uvec feature_order;
    if (this->params.shuffle_feature_order) {
      feature_order = arma::randperm(p);
    } else {
      feature_order = arma::linspace<arma::uvec>(0, p - 1, p);
    }

    arma::uvec::const_iterator it = feature_order.begin();
    const arma::uvec::const_iterator it_end = feature_order.end();

    bool swap = false;

    for (; it != it_end; ++it) {
      const arma::uword row_ix = *it;

      // No need to swap on last row!
      COUT << "PSI iter: " << i << " Swapping row: " << row_ix << "\n";
      // TODO: This needs to properly update the active set.
      const auto row_fit = this->psi_row_fit(row_ix);
      swap = swap || std::get<0>(row_fit);
      if (swap) {
        for (arma::uword row : {row_ix, std::get<1>(row_fit)}) {
          // TODO: Do we need to update every diagonal? Only need to update
          // Only updates the two items that changed!
          this->R.col(row_ix) -=
              this->theta(row_ix, row_ix) * this->Y.col(row_ix);
          this->theta(row_ix, row_ix) =
              R_nl(this->S_diag(row_ix),
                   arma::dot(this->R.col(row_ix), this->R.col(row_ix)));
          this->R.col(row_ix) +=
              this->theta(row_ix, row_ix) * this->Y.col(row_ix);
        }
        break;
      }
    }

    if (!swap) {
      break;
    } else {
      COUT << "PSI iter: " << i
           << " Post Swap cost: " << this->compute_objective() << " \n";
      static_cast<void>(this->fit());
      COUT << "PSI iter: " << i
           << " Post Swap Fit cost: " << this->compute_objective() << " \n";
    }
  }

  return fitmodel(this->theta, this->R, this->costs, this->active_set_size);
}

template <class TY, class TR, class TT, class TP>
std::tuple<bool, arma::uword>
CD<TY, TR, TT, TP>::psi_row_fit(const arma::uword row_ix) {
  COUT << "psi_row_fit row =  " << row_ix << " \n";
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));

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

  COUT << "selected super_active_set start =  " << *it << " \n";
  COUT << "selected super_active_set end =  " << *end << " \n";

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

  COUT << "zero_indices =  " << zeros << " \n";
  COUT << "non_zero_indices =  " << non_zeros << " \n";

  const arma::vec theta_diag = arma::vec(this->theta.diag());

  // TODO: Have parameter to shuffle order of iteration
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
    const arma::uword j = non_zero_order[*non_zeros_it];
    COUT << "Non Zero Index Loop: (" << row_ix << ", " << j << ") \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    R.col(row_ix) -= this->theta(row_ix, j) * this->Y.col(j);
    R.col(j) -= this->theta(j, row_ix) * this->Y.col(row_ix);
    this->theta(j, row_ix) = 0;
    this->theta(row_ix, j) = 0;

    const double aj = this->S_diag[row_ix] / theta_diag(j) +
                      this->S_diag[j] / theta_diag(row_ix);
    const double bj =
        2 *
        ((arma::dot(this->Y.col(j), this->R.col(row_ix)) / theta_diag(row_ix)) +
         (arma::dot(this->Y.col(row_ix), this->R.col(j)) / theta_diag(j)));

    // COUT << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // const std::tuple<double, double> theta_f = this->params.oracle.Qobj(aj,
    // bj);
    const std::tuple<double, double> theta_f =
        this->params.oracle.Qobj(aj, bj, row_ix, j);

    const double theta_j = std::get<0>(theta_f);
    const double f = std::get<1>(theta_f);

    const arma::vec a_vec = (this->S_diag(row_ix) / theta_diag(zeros) +
                             this->S_diag(zeros) / theta_diag(row_ix));
    const arma::vec b_vec =
        2 *
        (((this->Y.cols(zeros).t() * this->R.col(row_ix)) /
          theta_diag(row_ix)) +
         ((this->R.cols(zeros).t() * this->Y.col(row_ix)) / theta_diag(zeros)));
    // COUT << "Non Zero Index Loop: " << j << " theta_diag " << theta_diag << "
    // \n"; COUT << "Non Zero Index Loop: " << j << " b_vec " << b_vec << " \n";
    // COUT << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj vec \n";

    // COUT << "params.oracle.Qobj(a_vec, b_vec, row_ix) \n";
    // COUT << "params.oracle.Qobj(" << a_vec.size() << ", " <<  b_vec.size() <<
    // ", " << row_ix << ")\n"; COUT << "zeros.size() = " << zeros.size()
    // <<"\n"; std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // const std::tuple<arma::vec, arma::vec> thetas_fs =
    // this->params.oracle.Qobj(a_vec, b_vec);

    const std::tuple<arma::vec, arma::vec> thetas_fs =
        this->params.oracle.Qobj(a_vec, b_vec, row_ix, zeros);
    const arma::vec thetas = std::get<0>(thetas_fs);
    const arma::vec fs = std::get<1>(thetas_fs);

    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // COUT << "After this = " << zeros.size() <<"\n";
    // COUT << "Non Zero Index Loop: " << j << " fs: " << fs << " \n";
    // COUT << "Non Zero Index Loop: " << j << " thetas: "<< thetas <<"\n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (f < fs.min()) {
      // COUT << "Non Zero Index Loop: " << j << "f <= fs.min()" << "\n";
      this->theta(row_ix, j) = theta_j;
      this->theta(j, row_ix) = theta_j;
      this->R.col(row_ix) += theta_j * this->Y.col(j);
      this->R.col(j) += theta_j * this->Y.col(row_ix);
      COUT << "No swap for (" << row_ix << ", " << j << ") \n";
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
      // COUT << "Non Zero Index Loop: " << j << " Else" << "\n";
      const auto ell = arma::index_min(fs);
      // COUT << "Non Zero Index Loop: " << j << " ell = " << ell << "\n";
      const auto k = zeros(ell);
      // COUT << "Non Zero Index Loop: " << j << " k = zeros(ell) = " << k <<
      // "\n";
      const auto k_theta = thetas(ell);
      // COUT << "Non Zero Index Loop: " << j << "k_theta = " << k_theta <<
      // "\n";
      this->theta(row_ix, k) = k_theta;
      this->theta(k, row_ix) = k_theta;
      this->R.col(row_ix) += k_theta * this->Y.col(k);
      this->R.col(k) += k_theta * this->Y.col(row_ix);

      // Find location to remove (row_ix, j) from active set!
      const coordinate row_ix_j = {row_ix, j};
      auto low_j = std::lower_bound(this->active_set.begin(),
                                    this->active_set.end(), row_ix_j);
      const std::size_t loc_j = low_j - this->active_set.begin();

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
      auto low_k = std::lower_bound(k_begin, k_end, row_ix_k);

      const std::size_t loc_k = low_k - this->active_set.begin();

      // START: Debug Printing statements
      const auto cur_active_set_size = this->active_set.size();

      COUT << "Swap for (" << row_ix << ", " << j << ") with (" << row_ix
           << ", " << k << ")\n";
      COUT << "AS before insert:\n";
      COUT << "AS size = " << cur_active_set_size << " \n";

      const auto min_loc_kj = std::min(loc_k, loc_j);
      const auto tmp = (long long)(min_loc_kj - 2);
      const long long start_print_index = std::max(tmp - 2, 0LL);

      const auto end_print_index =
          std::min(std::max(loc_k + 2, loc_j + 2), cur_active_set_size - 1);
      index_print(COUT, this->active_set.begin() + start_print_index,
                  this->active_set.begin() + end_print_index,
                  start_print_index);

      if (loc_k >= cur_active_set_size) {
        COUT << " AS has no loc element -> Swapped item " << row_ix_k
             << "Is the last item in AS\n";
      } else {
        COUT << " AS(loc)= " << this->active_set.at(loc_k) << "\n";
        if (loc_k + 1 >= cur_active_set_size) {
          COUT << " AS has no loc+1 element\n";
        } else {
          COUT << " AS(loc+1)= " << this->active_set.at(loc_k + 1) << "\n";
        }
      }
      // END: Debug Printing statements

      // if (this->active_set.at(loc_k - 1) != row_ix_k){
      this->active_set.insert(low_k, row_ix_k);
      //}

      return {true, k};
    }
  }

  return {false, -1L};
}

template <class TY, class TR, class TT, class TP>
bool inline CD<TY, TR, TT, TP>::converged(const double old_objective,
                                          const double cur_objective,
                                          const size_t cur_iter) {
  return ((cur_iter > 1) &&
          (relative_gap(old_objective, cur_objective, this->params.gap_method,
                        this->params.one_normalize) <= this->params.tol));
}

template <class TY, class TR, class TT, class TP>
double inline CD<TY, TR, TT, TP>::compute_objective() {
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
std::tuple<double, double>
CD<TY, TR, TT, TP>::calc_ab(std::tuple<arma::uword, arma::uword> ij,
                            const arma::vec &theta_diag) {
  const arma::uword i = std::get<0>(ij);
  const arma::uword j = std::get<1>(ij);

  const double a =
      this->S_diag[i] / theta_diag(j) + this->S_diag[j] / theta_diag(i);
  const double b =
      2 * ((arma::dot(this->Y.col(j), this->R.col(i)) / theta_diag(i)) +
           (arma::dot(this->Y.col(i), this->R.col(j)) / theta_diag(j)));
  return {a, b};
}

template <class TY, class TR, class TT, class TP>
std::tuple<arma::vec, arma::vec>
CD<TY, TR, TT, TP>::calc_ab(const arma::uword i, const arma::uvec js,
                            const arma::vec &theta_diag) {
  const arma::vec a_vec =
      (this->S_diag(i) / theta_diag(js) + this->S_diag(js) / theta_diag(i));
  const arma::vec b_vec =
      2 * (((this->Y.cols(js).t() * this->R.col(i)) / theta_diag(i)) +
           ((this->R.cols(js).t() * this->Y.col(i)) / theta_diag(js)));
  return {a_vec, b_vec};
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::remove_from_support(
    std::tuple<arma::uword, arma::uword> ij) {
  const arma::uword i = std::get<0>(ij);
  const arma::uword j = std::get<1>(ij);

  R.col(i) -= this->theta(i, j) * this->Y.unsafe_col(j);
  R.col(j) -= this->theta(j, i) * this->Y.unsafe_col(i);
  this->theta(j, j) = 0;
  this->theta(j, j) = 0;
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::add_to_support(std::tuple<arma::uword, arma::uword> ij,
                                        const double theta) {
  const arma::uword i = std::get<0>(ij);
  const arma::uword j = std::get<1>(ij);

  R.col(i) += theta * this->Y.unsafe_col(j);
  R.col(j) += theta * this->Y.unsafe_col(i);
  this->theta(i, j) = theta;
  this->theta(j, i) = theta;
}

#endif // CD_H