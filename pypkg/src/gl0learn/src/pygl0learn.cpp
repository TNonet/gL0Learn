#include "arma_includes.h"
#include <pybind11/pybind11.h>

#include "pyfit.h"
#include "pyfitmodel.h"
#include "pyoracle.h"

namespace py = pybind11;

arma::umat upper_triangular_coords(const arma::uword p) {
  return coordinate_matrix_from_vector(upper_triangle_coordinate_vector(p));
}

double compute_objective(const arma::mat &theta, const arma::mat &residuals) {
  return objective(theta, residuals);
}

PYBIND11_MODULE(gl0learn_core, m) {
  m.attr("__name__") =
      "gl0learn.gl0learn_core"; // The default would be just "foo"
  m.def("union_of_correlated_features2",
        &union_of_correlated_features2<arma::mat>,
        R"pbdoc(
          Example function performing OLS.
          Parameters
          ----------
          arr : np.array
              input array
          Returns
          -------
          coeffs: np.ndarray
              coefficients
          std_err : np.ndarray
              standard error on the coefficients
      )pbdoc",
        py::arg("x"), py::arg("threshold"));

  init_oracle(m);
  init_fitmodel(m);
  init_fit(m);

  m.def("check_coordinate_matrix", &check_coordinate_matrix);
  m.def("check_is_coordinate_subset", &check_is_coordinate_subset,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
  m.def("upper_triangular_coords", &upper_triangular_coords);
  m.def("objective", &compute_objective);
  m.def("coordinate_vector_from_matrix", &coordinate_vector_from_matrix);
  m.def("coordinate_matrix_from_vector", &coordinate_matrix_from_vector);
}