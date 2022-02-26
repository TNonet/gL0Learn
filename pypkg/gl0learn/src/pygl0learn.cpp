#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "arma_includes.h"

#include "pyfitmodel.h"
#include "pyfit.h"
#include "pyoracle.h"

namespace py = pybind11;

arma::umat upper_triangular_coords(const arma::uword p){
    return coordinate_matrix_from_vector(upper_triangle_coordinate_vector(p));
}

PYBIND11_MODULE(_gl0learn, m) {
    m.attr("__name__") = "gl0learn._gl0learn"; // The default would be just "foo"
    m.def(
      "union_of_correlated_features2",
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
      py::arg("x"),
      py::arg("threshold")
    );

   init_oracle(m);
   init_fitmodel(m);
   init_fit(m);

   m.def("check_coordinate_matrix", &check_coordinate_matrix);
   m.def("check_is_coordinate_subset", &check_is_coordinate_subset);
   m.def("upper_triangular_coords", &upper_triangular_coords);
}