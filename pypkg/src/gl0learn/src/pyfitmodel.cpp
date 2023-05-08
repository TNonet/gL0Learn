#include "pyfitmodel.h"

void init_fitmodel(py::module_ &m) {
  py::class_<fitmodel>(m, "_fitmodel")
      .def(py::init<const arma::mat &, const arma::mat &,
                    const std::vector<double> &,
                    const std::vector<std::size_t> >())
      .def(py::init<fitmodel const &>())
      .def_readonly("theta", &fitmodel::theta)
      .def_readonly("R", &fitmodel::R)
      .def_readonly("costs", &fitmodel::costs)
      .def_readonly("active_set_size", &fitmodel::active_set_size);
}
