#include "pyoracle.h"

void init_oracle(py::module_ &m) {
  declare_bounds<double>(m, "double");
  declare_bounds<arma::mat>(m, "mat");
  declare_penalty_l0<double>(m, "double");
  declare_penalty_l0<arma::mat>(m, "mat");
  //  declare_penalty_l1<double>(m, "double");
  //  declare_penalty_l1<arma::mat>(m, "mat");
  //  declare_penalty_l2<double>(m, "double");
  //  declare_penalty_l2<arma::mat>(m, "mat");
  declare_penalty_l0l1<double>(m, "double");
  declare_penalty_l0l1<arma::mat>(m, "mat");
  declare_penalty_l0l2<double>(m, "double");
  declare_penalty_l0l2<arma::mat>(m, "mat");
  declare_penalty_l0l1l2<double>(m, "double");
  declare_penalty_l0l1l2<arma::mat>(m, "mat");

  py::class_<NoBounds>(m, "_NoBounds")
      .def(py::init())
      .def("validate", &NoBounds::validate);
}
