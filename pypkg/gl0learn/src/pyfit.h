#ifndef PY_FIT_H_
#define PY_FIT_H_
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "arma_includes.h"
#include "gL0Learn.h"
#include "oracle.h"
#include "fitmodel.h"


namespace py = pybind11;

template<typename P, typename B>
const fitmodel _fit(const arma::mat& Y,
              const arma::mat& theta_init,
              const P& penalty,
              const B& bounds,
              const std::string algorithm,
              const arma::umat& initial_active_set,
              const arma::umat& super_active_set,
              const double atol,
              const double rtol,
              const size_t max_active_set_size,
              const size_t max_iter){

    const Oracle<P, B> oracle = Oracle<P, B>(penalty, bounds);

    return gL0LearnFit(Y, theta_init, oracle, algorithm, initial_active_set, super_active_set, atol, rtol,
                       max_active_set_size, max_iter);
}

void init_fit(py::module_ &m) {

   m.def("_fit", &_fit<PenaltyL0<double>, NoBounds>,
         py::call_guard<py::scoped_ostream_redirect,
         py::scoped_estream_redirect>());
   m.def("_fit", &_fit<PenaltyL0L2<double>, NoBounds>,
         py::call_guard<py::scoped_ostream_redirect,
         py::scoped_estream_redirect>());

}

#endif // PY_FIT_H_