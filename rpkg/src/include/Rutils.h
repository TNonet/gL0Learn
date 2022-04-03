#ifndef R_UTILS_H_
#define R_UTILS_H_
#include "arma_includes.h"

inline bool is_double_SEXP(const SEXP &x) {
  // {is.atomic(x) && length(x) == 1L && !is.character(x)
  return Rf_isVectorAtomic(x) && Rf_isReal(x) && Rf_length(x) == 1L;
}

#endif // R_UTILS_H_