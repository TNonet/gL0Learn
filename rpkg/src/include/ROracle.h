#ifndef R_ORACLE_H
#define R_ORACLE_H
#include <string>

#include "arma_includes.h"
#include "fitmodel.h"
#include "gL0Learn.h"
#include "oracle.h"

template <typename T>
void declare_bounds(const std::string &typestr) {
  using Bounds_ = Bounds<T>;
  std::string RBoundsName = std::string("Bounds_") + typestr;

  Rcpp::class_<Bounds_>(RBoundsName.c_str())
      .template constructor<T, T>()
      .field_readonly("lows", &Bounds_::lows)
      .field_readonly("highs", &Bounds_::highs)
      .method("validate", &Bounds_::validate);
}

template <typename P>
struct WrappedPenalty : public P {
  using P::P;
  double const compute_objective_penalty(const arma::mat &theta,
                                         const arma::mat &residuals,
                                         const arma::umat &active_set_mat) {
    const auto active_set = coordinate_vector_from_matrix(active_set_mat);
    return objective(theta, residuals, active_set, *this);
  }

  // TOOD: Conditionally include based on testing macro
  template <typename B>
  arma::mat const prox(const arma::mat &theta, const B &bounds) {
    const P unwrappedPenalty = *this;
    const auto oracle = Oracle<P, B>(unwrappedPenalty, bounds);
    return oracle.prox(theta);
  }

  template <typename B>
  Rcpp::List const fit(const arma::mat &Y, const arma::mat &theta_init,
                       const std::string &algorithm, const B &bounds,
                       const arma::umat &initial_active_set,
                       const arma::umat &super_active_set, const double tol,
                       const size_t max_active_set_size, const size_t max_iter,
                       const size_t seed, const size_t max_swaps,
                       const bool shuffle_feature_order) {
    const P unwrappedPenalty = *this;
    const auto oracle = Oracle<P, B>(unwrappedPenalty, bounds);

    const auto l =
        gL0LearnFit(Y, theta_init, oracle, algorithm, initial_active_set,
                    super_active_set, tol, max_active_set_size, max_iter, seed,
                    max_swaps, shuffle_feature_order);

    return Rcpp::List::create(
        Rcpp::Named("theta") = l.theta, Rcpp::Named("R") = l.R,
        Rcpp::Named("costs") = l.costs,
        Rcpp::Named("active_set_size") = l.active_set_size);
  }
};

template <typename T, typename P, typename W>
auto declare_wrapped_penalty(const std::string &RPenaltyClassName,
                             const std::string &RWrappedPenaltyClassName) {
  const auto Rclass =
      Rcpp::class_<W>(RWrappedPenaltyClassName.c_str())
          .template derives<P>(RPenaltyClassName.c_str())
          .method("objective", &W::compute_objective_penalty)
          .method("fit_", &W::template fit<NoBounds>)
          .method("fit_double", &W::template fit<Bounds<double> >)
          .method("fit_mat", &W::template fit<Bounds<arma::mat> >)
          .method("prox_mat_", &W::template prox<NoBounds>)
          .method("prox_mat_double", &W::template prox<Bounds<double> >)
          .method("prox_mat_mat", &W::template prox<Bounds<arma::mat> >);

  // .method("prox_double_" &W::template prox<double, NoBounds>)
  // .method("prox_double_double" &W::template prox<double, Bounds<double>>)
  // .method("prox_double_mat" &W::template prox<double, Bounds<arma::mat>>)

  // TODO(TNonet) Conditionally include based on testing macro
  return Rclass;
}

template <typename T>
void declare_penalty(const std::string &typestr) {
  using PenaltyL0_ = PenaltyL0<T>;
  using WrappedPenaltyL0_ = WrappedPenalty<PenaltyL0_>;
  std::string RPenaltyClassNameL0 = std::string("PenaltyL0_") + typestr;
  std::string RWrappedPenaltyClassNameL0 =
      std::string("WrappedPenaltyL0_") + typestr;

  Rcpp::class_<PenaltyL0_>(RPenaltyClassNameL0.c_str())
      .template constructor<T>()
      .field_readonly("l0", &PenaltyL0_::l0)
      .method("validate", &PenaltyL0_::validate);

  declare_wrapped_penalty<T, PenaltyL0_, WrappedPenaltyL0_>(
      RPenaltyClassNameL0, RWrappedPenaltyClassNameL0)
      .template constructor<T>();

  using PenaltyL1_ = PenaltyL1<T>;
  // using WrappedPenaltyL1_ = WrappedPenalty<PenaltyL1_>;
  std::string RPenaltyClassNameL1 = std::string("PenaltyL1_") + typestr;
  // std::string RWrappedPenaltyClassNameL1 =
  //     std::string("WrappedPenaltyL1_") + typestr;

  Rcpp::class_<PenaltyL1_>(RPenaltyClassNameL1.c_str())
      .template constructor<T>()
      .field_readonly("l1", &PenaltyL1_::l1)
      .method("validate", &PenaltyL1_::validate);

  // declare_wrapped_penalty<T, PenaltyL1_,
  // WrappedPenaltyL1_>(RPenaltyClassNameL1, RWrappedPenaltyClassNameL1);

  using PenaltyL2_ = PenaltyL2<T>;
  // using WrappedPenaltyL2_ = WrappedPenalty<PenaltyL2_>;
  std::string RPenaltyClassNameL2 = std::string("PenaltyL2_") + typestr;
  // std::string RWrappedPenaltyClassNameL2 =
  //     std::string("WrappedPenaltyL2_") + typestr;

  Rcpp::class_<PenaltyL2_>(RPenaltyClassNameL2.c_str())
      .template constructor<T>()
      .field_readonly("l2", &PenaltyL2_::l2)
      .method("validate", &PenaltyL2_::validate);

  // declare_wrapped_penalty<T, PenaltyL2_,
  // WrappedPenaltyL2_>(RPenaltyClassNameL2, RWrappedPenaltyClassNameL2);

  // using PenaltyL0L1_ = PenaltyL0L1<T>;
  // using WrappedPenaltyL0L1_ = WrappedPenalty<PenaltyL0L1_>;
  // std::string RPenaltyClassNameL0L1 = std::string("PenaltyL0L1_") + typestr;
  // std::string RWrappedPenaltyClassNameL0L1 =
  //     std::string("WrappedPenaltyL0L1_") + typestr;
  //
  // Rcpp::class_<PenaltyL0L1_>(RPenaltyClassNameL0L1.c_str())
  //     .template derives<PenaltyL0_>(RPenaltyClassNameL0.c_str())
  //     .template derives<PenaltyL1_>(RPenaltyClassNameL1.c_str())
  //     .method("validate", &PenaltyL0L1_::validate);
  //
  // declare_wrapped_penalty<T, PenaltyL0L1_,
  // WrappedPenaltyL0L1_>(RPenaltyClassNameL0L1, RWrappedPenaltyClassNameL0L1);

  using PenaltyL0L2_ = PenaltyL0L2<T>;
  using WrappedPenaltyL0L2_ = WrappedPenalty<PenaltyL0L2_>;
  std::string RPenaltyClassNameL0L2 = std::string("PenaltyL0L2_") + typestr;
  std::string RWrappedPenaltyClassNameL0L2 =
      std::string("WrappedPenaltyL0L2_") + typestr;

  Rcpp::class_<PenaltyL0L2_>(RPenaltyClassNameL0L2.c_str())
      .template constructor<T, T>()
      .template derives<PenaltyL0_>(RPenaltyClassNameL0.c_str())
      .template derives<PenaltyL2_>(RPenaltyClassNameL1.c_str())
      .method("validate", &PenaltyL0L2_::validate);

  declare_wrapped_penalty<T, PenaltyL0L2_, WrappedPenaltyL0L2_>(
      RPenaltyClassNameL0L2, RWrappedPenaltyClassNameL0L2)
      .template constructor<T, T>();

  using PenaltyL0L1L2_ = PenaltyL0L1L2<T>;
  using WrappedPenaltyL0L1L2_ = WrappedPenalty<PenaltyL0L1L2_>;
  std::string RPenaltyClassNameL0L1L2 = std::string("PenaltyL0L1L2_") + typestr;
  std::string RWrappedPenaltyClassNameL0L1L2 =
      std::string("WrappedPenaltyL0L1L2_") + typestr;

  Rcpp::class_<PenaltyL0L1L2_>(RPenaltyClassNameL0L1L2.c_str())
      .template constructor<T, T, T>()
      .template derives<PenaltyL0_>(RPenaltyClassNameL0.c_str())
      .template derives<PenaltyL1_>(RPenaltyClassNameL1.c_str())
      .template derives<PenaltyL2_>(RPenaltyClassNameL2.c_str())
      .method("validate", &PenaltyL0L1L2_::validate);

  declare_wrapped_penalty<T, PenaltyL0L1L2_, WrappedPenaltyL0L1L2_>(
      RPenaltyClassNameL0L1L2, RWrappedPenaltyClassNameL0L1L2)
      .template constructor<T, T, T>();
}

#endif  // R_ORACLE_H
