#ifndef PY_FITMODEL_H_
#define PY_FITMODEL_H_
#include "arma_includes.h"
#include "fitmodel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_fitmodel(py::module_ &m) {
  py::class_<fitmodel>(m, "_fitmodel")
      .def(py::init<const arma::mat &, const arma::mat &,
                    const std::vector<double> &,
                    const std::vector<std::size_t>>())
      .def(py::init<fitmodel const &>())
      .def_readonly("theta", &fitmodel::theta)
      .def_readonly("R", &fitmodel::R)
      .def_readonly("costs", &fitmodel::costs)
      .def_readonly("active_set_size", &fitmodel::active_set_size);
}

#endif // PY_FITMODEL_H_
