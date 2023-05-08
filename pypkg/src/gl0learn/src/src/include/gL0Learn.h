#ifndef GL0LEARN_H
#define GL0LEARN_H
#include <string>

#include "CD.h"
#include "arma_includes.h"
#include "fitmodel.h"
#include "oracle.h"

template <class T, class O>
fitmodel gL0LearnFit(const T &Y, const T &theta_init, const O &oracle,
                     const std::string &algorithm,
                     const arma::umat &initial_active_set,
                     const arma::umat &super_active_set, const double tol,
                     const size_t max_active_set_size, const size_t max_iter,
                     const size_t seed, const size_t max_swaps,
                     const bool shuffle_feature_order) {
  arma::arma_rng::set_seed(seed);

  const auto params =
      CDParams<O>(tol, max_active_set_size, GapMethod::both, true, max_iter,
                  oracle, algorithm, max_swaps, shuffle_feature_order);

  const coordinate_vector initial_active_set_vec =
      coordinate_vector_from_matrix(initial_active_set);

  const coordinate_vector super_active_set_vec =
      coordinate_vector_from_matrix(super_active_set);

  auto cd = CD<const T, T, T, CDParams<O> >(
      Y, theta_init, params, initial_active_set_vec, super_active_set_vec);

  if (algorithm == "CD") {
    return cd.fit();
  } else if (algorithm == "CDPSI") {
    return cd.fitpsi();
  } else {
    STOP("Canno't determine algorithm choice");
  }
}

#endif  // GL0LEARN_H
