#include "pyfit.h"

void init_fit(py::module_ &m) {
  m.def("_fit", &fit<WrappedPenalty<PenaltyL0<double>>, NoBounds>,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
  m.def("_fit", &fit<WrappedPenalty<PenaltyL0L1<double>>, NoBounds>,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
  m.def("_fit", &fit<WrappedPenalty<PenaltyL0L2<double>>, NoBounds>,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
  m.def("_fit", &fit<WrappedPenalty<PenaltyL0L1L2<double>>, NoBounds>,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>());
}
