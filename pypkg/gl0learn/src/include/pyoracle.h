#ifndef PY_ORACLE_H_
#define PY_ORACLE_H_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "arma_includes.h"
#include "oracle.h"

namespace py = pybind11;

void init_oracle(py::module_ &m);

template <template <typename ...> class W, typename P>
P unwrapped(const W<P> &wrappedPenalty){
  const P& unwrappedPenalty = wrappedPenalty;
  return unwrappedPenalty;
}

template <typename P>
struct WrappedPenalty : public P {
  using P::P;

  double objective_(const arma::mat &theta,
                    const arma::mat &residuals) {
    return objective(theta, residuals, unwrapped(*this));
  }

  double objective_from_active_set_mat(const arma::mat &theta,
                                       const arma::mat &residuals,
                                       const arma::umat &active_set_mat) {
    const auto active_set = coordinate_vector_from_matrix(active_set_mat);
    return objective(theta, residuals, active_set, unwrapped(*this));
  }

  double objective_from_active_set(const arma::mat &theta,
                                   const arma::mat &residuals,
                                   const coordinate_vector &active_set) {
    return objective(theta, residuals, active_set, *this);
  }

  // TOOD: Conditionally include based on testing macro
  template <typename B>
  arma::mat prox(const arma::mat &theta, const B &bounds) {
    const auto oracle = Oracle<P, B>(unwrapped(*this), bounds);
    return oracle.prox(theta);
  }

  double penalty_cost_(const arma::mat &theta) {
    return penalty_cost(theta, unwrapped(*this));
  }

  double penalty_cost_from_active_set(const arma::mat &theta,
                                      const arma::umat &active_set_mat) {
    const auto active_set = coordinate_vector_from_matrix(active_set_mat);
    return penalty_cost(theta, active_set, unwrapped(*this));
  }
};

template <typename T>
void declare_bounds(py::module &m, const std::string &typestr) {
  using Bounds_ = Bounds<T>;
  std::string pyclass_name = std::string("_Bounds_") + typestr;
  py::class_<Bounds_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                      py::dynamic_attr())
      .def(py::init<const T, const T>())
      .def_readonly("lows", &Bounds_::lows)
      .def_readonly("highs", &Bounds_::highs)
      .def("validate", &Bounds_::validate);
}

template <class W, class P>
void declare_penalty_common(P &penalty_py_class){
  penalty_py_class
      .def("objective_from_active_set_mat", &W::objective_from_active_set_mat)
      .def("objective_from_active_set", &W::objective_from_active_set)
      .def("objective", &W::objective_)
      .def("penalty_cost", &W::penalty_cost_)
      .def("penalty_cost_from_active_set", &W::penalty_cost_from_active_set)
      .def("prox_mat_", &W::template prox<NoBounds>)
      .def("prox_mat_double", &W::template prox<Bounds<double> >)
      .def("prox_mat_mat", &W::template prox<Bounds<arma::mat> >);

}

template <class T>
void declare_penalty_l0(py::module &m, const std::string &typestr) {
  using Penalty_ = WrappedPenalty<PenaltyL0<T>>;
  std::string pyclass_name = std::string("_PenaltyL0_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def("validate", &Penalty_::validate);
  declare_penalty_common<Penalty_>(py_class);
}

//template <class T>
//void declare_penalty_l1(py::module &m, const std::string &typestr) {
//  using Penalty_ = WrappedPenalty<PenaltyL1<T>>;
//  std::string pyclass_name = std::string("_PenaltyL1_") + typestr;
//  auto py_class =
//      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
//                           py::dynamic_attr())
//          .def(py::init<const T>())
//          .def_readonly("l1", &Penalty_::l1)
//          .def("validate", &Penalty_::validate);
//  declare_penalty_common<Penalty_>(py_class);
//}
//
//template <class T>
//void declare_penalty_l2(py::module &m, const std::string &typestr) {
//  using Penalty_ = WrappedPenalty<PenaltyL2<T>>;
//  std::string pyclass_name = std::string("_PenaltyL2_") + typestr;
//  auto py_class =
//      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
//                           py::dynamic_attr())
//          .def(py::init<const T>())
//          .def_readonly("l2", &Penalty_::l2)
//          .def("validate", &Penalty_::validate);
//  declare_penalty_common<Penalty_>(py_class);
//}

template <class T>
void declare_penalty_l0l1(py::module &m, const std::string &typestr) {
  using Penalty_ = WrappedPenalty<PenaltyL0L1<T>>;
  std::string pyclass_name = std::string("_PenaltyL0L1_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l1", &Penalty_::l1)
          .def("validate", &Penalty_::validate);
  declare_penalty_common<Penalty_>(py_class);
}

template <class T>
void declare_penalty_l0l2(py::module &m, const std::string &typestr) {
  using Penalty_ = WrappedPenalty<PenaltyL0L2<T>>;
  std::string pyclass_name = std::string("_PenaltyL0L2_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l2", &Penalty_::l2)
          .def("validate", &Penalty_::validate);
  declare_penalty_common<Penalty_>(py_class);
}

template <class T>
void declare_penalty_l0l1l2(py::module &m, const std::string &typestr) {
  using Penalty_ = WrappedPenalty<PenaltyL0L1L2<T>>;
  std::string pyclass_name = std::string("_PenaltyL0L1L2_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l1", &Penalty_::l1)
          .def_readonly("l2", &Penalty_::l2)
          .def("validate", &Penalty_::validate);
  declare_penalty_common<Penalty_>(py_class);
}

template void declare_bounds<double>(py::module &m, const std::string &typestr);
template void declare_bounds<arma::mat>(py::module &m,
                                        const std::string &typestr);
template void declare_penalty_l0<double>(py::module &m,
                                         const std::string &typestr);
template void declare_penalty_l0<arma::mat>(py::module &m,
                                            const std::string &typestr);
//template void declare_penalty_l1<double>(py::module &m,
//                                         const std::string &typestr);
//template void declare_penalty_l1<arma::mat>(py::module &m,
//                                            const std::string &typestr);
//template void declare_penalty_l2<double>(py::module &m,
//                                         const std::string &typestr);
//template void declare_penalty_l2<arma::mat>(py::module &m,
//                                            const std::string &typestr);
template void declare_penalty_l0l1<double>(py::module &m,
                                           const std::string &typestr);
template void declare_penalty_l0l1<arma::mat>(py::module &m,
                                              const std::string &typestr);
template void declare_penalty_l0l2<double>(py::module &m,
                                           const std::string &typestr);
template void declare_penalty_l0l2<arma::mat>(py::module &m,
                                              const std::string &typestr);
template void declare_penalty_l0l1l2<double>(py::module &m,
                                             const std::string &typestr);
template void declare_penalty_l0l1l2<arma::mat>(py::module &m,
                                                const std::string &typestr);

#endif  // PY_ORACLE_H_
