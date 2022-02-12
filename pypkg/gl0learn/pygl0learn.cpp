#include <pybind11/pybind11.h>
#include "utils.h"
#include "active_set.h"


arma::umat test_union_of_correlated_features2(const arma::mat & x,
                                              const double threshold){
    return union_of_correlated_features2(x, threshold);
}


namespace py = pybind11;

PYBIND11_MODULE(gl0learn, m) {
    m.def(
      "test_union_of_correlated_features2",
      &test_union_of_correlated_features2,
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
}