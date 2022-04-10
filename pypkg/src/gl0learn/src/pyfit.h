#ifndef PY_FIT_H_
#define PY_FIT_H_
#include "arma_includes.h"
#include "fitmodel.h"
#include "gL0Learn.h"
#include "oracle.h"
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_fit(py::module_ &m);

template <typename P, typename B>
fitmodel fit(const arma::mat &Y, const arma::mat &theta_init, const P &penalty,
             const B &bounds, const std::string algorithm,
             const arma::umat &initial_active_set,
             const arma::umat &super_active_set, const double tol,
             const size_t max_active_set_size, const size_t max_iter,
             const size_t seed, const size_t max_swaps,
             const bool shuffle_feature_order) {

  const Oracle<P, B> oracle = Oracle<P, B>(penalty, bounds);

  return gL0LearnFit(Y, theta_init, oracle, algorithm, initial_active_set,
                     super_active_set, tol, max_active_set_size, max_iter, seed,
                     max_swaps, shuffle_feature_order);
}

template fitmodel fit<PenaltyL0<double>, NoBounds>(
    const arma::mat &Y, const arma::mat &theta_init,
    const PenaltyL0<double> &penalty, const NoBounds &bounds,
    const std::string algorithm, const arma::umat &initial_active_set,
    const arma::umat &super_active_set, const double tol,
    const size_t max_active_set_size, const size_t max_iter, const size_t seed,
    const size_t max_swaps, const bool shuffle_feature_order);

template fitmodel fit<PenaltyL0L1<double>, NoBounds>(
    const arma::mat &Y, const arma::mat &theta_init,
    const PenaltyL0L1<double> &penalty, const NoBounds &bounds,
    const std::string algorithm, const arma::umat &initial_active_set,
    const arma::umat &super_active_set, const double tol,
    const size_t max_active_set_size, const size_t max_iter, const size_t seed,
    const size_t max_swaps, const bool shuffle_feature_order);

template fitmodel fit<PenaltyL0L2<double>, NoBounds>(
    const arma::mat &Y, const arma::mat &theta_init,
    const PenaltyL0L2<double> &penalty, const NoBounds &bounds,
    const std::string algorithm, const arma::umat &initial_active_set,
    const arma::umat &super_active_set, const double tol,
    const size_t max_active_set_size, const size_t max_iter, const size_t seed,
    const size_t max_swaps, const bool shuffle_feature_order);

template fitmodel fit<PenaltyL0L1L2<double>, NoBounds>(
    const arma::mat &Y, const arma::mat &theta_init,
    const PenaltyL0L1L2<double> &penalty, const NoBounds &bounds,
    const std::string algorithm, const arma::umat &initial_active_set,
    const arma::umat &super_active_set, const double tol,
    const size_t max_active_set_size, const size_t max_iter, const size_t seed,
    const size_t max_swaps, const bool shuffle_feature_order);

#endif // PY_FIT_H_