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

template <class P, class T>
void declare_penalty_cost(T &penalty_py_class) {
  penalty_py_class
      .def("cost", &P::template cost<arma::mat, arma::uword, arma::uword>)
      .def("cost", &P::template cost<arma::mat, arma::uword>)
      .def("cost", &P::template cost<arma::mat>)
      .def("cost", &P::template cost<double, arma::uword, arma::uword>)
      .def("cost", &P::template cost<double, arma::uword>)
      .def("cost", &P::template cost<double>);
}

template <class P>
double compute_objective_penalty(const arma::mat &theta,
                                 const arma::mat &residuals,
                                 const arma::umat &active_set_mat,
                                 const P &penalty) {
  const auto active_set = coordinate_vector_from_matrix(active_set_mat);
  return objective(theta, residuals, active_set, penalty);
}

template <class T>
void declare_penalty_l0(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL0<T>;
  std::string pyclass_name = std::string("_PenaltyL0_") + typestr;

  auto f1 = &Penalty_::template cost<arma::mat, arma::uword, arma::uword>;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def("validate", &Penalty_::validate)
          .def("cost", f1)
          .def("cost", &Penalty_::template cost<arma::mat, arma::uword>)
          .def("cost", &Penalty_::template cost<arma::mat>)
          .def("cost",
               &Penalty_::template cost<double, arma::uword, arma::uword>)
          .def("cost", &Penalty_::template cost<double, arma::uword>)
          .def("cost", &Penalty_::template cost<double>);
  // declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template <class T>
void declare_penalty_l1(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL1<T>;
  std::string pyclass_name = std::string("_PenaltyL1_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T>())
          .def_readonly("l1", &Penalty_::l1)
          .def("validate", &Penalty_::validate);
  declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template <class T>
void declare_penalty_l2(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL2<T>;
  std::string pyclass_name = std::string("_PenaltyL2_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T>())
          .def_readonly("l2", &Penalty_::l2)
          .def("validate", &Penalty_::validate);
  declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template <class T>
void declare_penalty_l0l1(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL0L1<T>;
  std::string pyclass_name = std::string("_PenaltyL0L1_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l1", &Penalty_::l1)
          .def("validate", &Penalty_::validate);
  declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template <class T>
void declare_penalty_l0l2(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL0L2<T>;
  std::string pyclass_name = std::string("_PenaltyL0L2_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l2", &Penalty_::l2)
          .def("validate", &Penalty_::validate);
  declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template <class T>
void declare_penalty_l0l1l2(py::module &m, const std::string &typestr) {
  using Penalty_ = PenaltyL0L1L2<T>;
  std::string pyclass_name = std::string("_PenaltyL0L1L2_") + typestr;
  auto py_class =
      py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(),
                           py::dynamic_attr())
          .def(py::init<const T, const T, const T>())
          .def_readonly("l0", &Penalty_::l0)
          .def_readonly("l1", &Penalty_::l1)
          .def_readonly("l2", &Penalty_::l2)
          .def("validate", &Penalty_::validate);
  declare_penalty_cost<Penalty_>(py_class);
  m.def("objective", &compute_objective_penalty<Penalty_>);
}

template void declare_bounds<double>(py::module &m, const std::string &typestr);
template void declare_bounds<arma::mat>(py::module &m,
                                        const std::string &typestr);
template void declare_penalty_l0<double>(py::module &m,
                                         const std::string &typestr);
template void declare_penalty_l0<arma::mat>(py::module &m,
                                            const std::string &typestr);
template void declare_penalty_l1<double>(py::module &m,
                                         const std::string &typestr);
template void declare_penalty_l1<arma::mat>(py::module &m,
                                            const std::string &typestr);
template void declare_penalty_l2<double>(py::module &m,
                                         const std::string &typestr);
template void declare_penalty_l2<arma::mat>(py::module &m,
                                            const std::string &typestr);
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
