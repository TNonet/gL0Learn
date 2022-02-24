#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "gL0Learn.h"
#include "utils.h"
#include "active_set.h"
#include "oracle.h"
#include "fitmodel.h"

#include <chrono>
#include <thread>

namespace py = pybind11;

arma::umat upper_triangular_coords(const arma::uword p){
    return coordinate_matrix_from_vector(upper_triangle_coordinate_vector(p));
}

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


    COUT << "_fit \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const Oracle<P, B> oracle = Oracle<P, B>(penalty, bounds);

    return gL0LearnFit(Y, theta_init, oracle, algorithm, initial_active_set, super_active_set, atol, rtol,
                       max_active_set_size, max_iter);
}

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

PYBIND11_MODULE(_gl0learn, m) {
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
    declare_bounds<double>(m, "double");
    declare_bounds<arma::mat>(m, "mat");
    declare_penalty_l0<double>(m, "double");
    declare_penalty_l0<arma::mat>(m, "mat");
    declare_penalty_l1<double>(m, "double");
    declare_penalty_l1<arma::mat>(m, "mat");
    declare_penalty_l2<double>(m, "double");
    declare_penalty_l2<arma::mat>(m, "mat");
    declare_penalty_l0l1<double>(m, "double");
    declare_penalty_l0l1<arma::mat>(m, "mat");
    declare_penalty_l0l2<double>(m, "double");
    declare_penalty_l0l2<arma::mat>(m, "mat");
    declare_penalty_l0l1l2<double>(m, "double");
    declare_penalty_l0l1l2<arma::mat>(m, "mat");

    py::class_<NoBounds>(m, "_NoBounds")
        .def(py::init())
        .def("validate", &NoBounds::validate);

   py::class_<fitmodel>(m, "_fitmodel")
       .def(py::init())
       .def(py::init<const arma::mat&,
                     const arma::mat&,
                     const std::vector<double>&,
                     const std::vector<std::size_t>>())
       .def(py::init<fitmodel const &>())
       .def_readonly("theta", &fitmodel::theta)
       .def_readonly("R", &fitmodel::R)
       .def_readonly("costs", &fitmodel::costs)
       .def_readonly("active_set_size", &fitmodel::active_set_size);

   m.def("_fit", &_fit<PenaltyL0<double>, NoBounds>,
         py::call_guard<py::scoped_ostream_redirect,
         py::scoped_estream_redirect>());
   m.def("_fit", &_fit<PenaltyL0L2<double>, NoBounds>,
         py::call_guard<py::scoped_ostream_redirect,
         py::scoped_estream_redirect>());
   m.def("check_coordinate_matrix", &check_coordinate_matrix);
   m.def("check_is_coordinate_subset", &check_is_coordinate_subset);
   m.def("upper_triangular_coords", &upper_triangular_coords);
}