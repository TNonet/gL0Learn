#ifndef PY_FITMODEL_H_
#define PY_FITMODEL_H_
#include "arma_includes.h"
#include "fitmodel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_fitmodel(py::module_ &m);

#endif // PY_FITMODEL_H_
