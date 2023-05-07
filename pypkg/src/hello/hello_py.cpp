#include <iostream>
#include <stdexcept>


#include <pybind11/pybind11.h>
#include <carma>
#include <armadillo>

namespace py = pybind11;

void hello() {
    std::cout << "Hello, World!" << std::endl;
}

int return_two() {
    return 2;
}


class MatrixHolder {
 public:
  explicit MatrixHolder(size_t d) {
    A = arma::Mat<double>(d, d, arma::fill::eye);
    std::cerr << "filled arma matrix\n";
  }
  arma::Mat<double> A;
};

PYBIND11_MODULE(_hello, m) {
    m.doc() = "_hello";
    m.def("hello", &hello, "Prints \"Hello, World!\"");
    m.def("return_two", &return_two, "Returns 2");

    py::class_<MatrixHolder>(m, "MH").def(py::init<size_t>())
      .def_readwrite("A", &MatrixHolder::A);
}