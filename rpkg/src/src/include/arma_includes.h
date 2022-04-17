#ifndef INCLUDES_H
#define INCLUDES_H

#include <exception>
#include <stdexcept>
#include <utility>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#define COUT Rcpp::Rcout
#define STOP Rcpp::stop

void inline UserInterrupt() { Rcpp::checkUserInterrupt(); }

#endif // INCLUDES_H