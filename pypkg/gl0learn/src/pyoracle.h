#ifndef PY_ORACLE_H_
#define PY_ORACLE_H_
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "arma_includes.h"
#include "oracle.h"

namespace py = pybind11;

template<typename T>
void declare_bounds(py::module &m, std::string typestr) {
    using Bounds_ = Bounds<T>;
    std::string pyclass_name = std::string("_Bounds_") + typestr;
    py::class_<Bounds_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T, const T>())
    .def_readonly("lows", &Bounds_::lows)
    .def_readonly("highs", &Bounds_::highs)
    .def("validate", &Bounds_::validate);
}

template<class T>
void declare_penalty_l0(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL0<T>;
    std::string pyclass_name = std::string("_PenaltyL0_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T>())
    .def_readonly("l0", &Penalty_::l0)
    .def("validate", &Penalty_::validate);
}

template<class T>
void declare_penalty_l1(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL1<T>;
    std::string pyclass_name = std::string("_PenaltyL1_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T>())
    .def_readonly("l1", &Penalty_::l1)
    .def("validate", &Penalty_::validate);
}

template<class T>
void declare_penalty_l2(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL2<T>;
    std::string pyclass_name = std::string("_PenaltyL2_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T>())
    .def_readonly("l2", &Penalty_::l2)
    .def("validate", &Penalty_::validate);
}

template<class T>
void declare_penalty_l0l1(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL0L1<T>;
    std::string pyclass_name = std::string("_PenaltyL0L1_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T, const T>())
    .def_readonly("l0", &Penalty_::l0)
    .def_readonly("l1", &Penalty_::l1)
    .def("validate", &Penalty_::validate);
}

template<class T>
void declare_penalty_l0l2(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL0L2<T>;
    std::string pyclass_name = std::string("_PenaltyL0L2_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T, const T>())
    .def_readonly("l0", &Penalty_::l0)
    .def_readonly("l2", &Penalty_::l2)
    .def("validate", &Penalty_::validate);
}

template<class T>
void declare_penalty_l0l1l2(py::module &m, std::string typestr) {
    using Penalty_ = PenaltyL0L1L2<T>;
    std::string pyclass_name = std::string("_PenaltyL0L1L2_") + typestr;
    py::class_<Penalty_>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const T, const T, const T>())
    .def_readonly("l0", &Penalty_::l0)
    .def_readonly("l1", &Penalty_::l1)
    .def_readonly("l2", &Penalty_::l2)
    .def("validate", &Penalty_::validate);
}

void init_oracle(py::module_ &m);

template void declare_bounds<double>(py::module &m, std::string typestr);
template void declare_bounds<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l0<double>(py::module &m, std::string typestr);
template void declare_penalty_l0<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l1<double>(py::module &m, std::string typestr);
template void declare_penalty_l1<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l2<double>(py::module &m, std::string typestr);
template void declare_penalty_l2<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l0l1<double>(py::module &m, std::string typestr);
template void declare_penalty_l0l1<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l0l2<double>(py::module &m, std::string typestr);
template void declare_penalty_l0l2<arma::mat>(py::module &m, std::string typestr);
template void declare_penalty_l0l1l2<double>(py::module &m, std::string typestr);
template void declare_penalty_l0l1l2<arma::mat>(py::module &m, std::string typestr);

#endif  // PY_ORACLE_H_