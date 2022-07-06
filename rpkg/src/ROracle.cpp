#include "ROracle.h"

// [[Rcpp::depends(RcppArmadillo)]]

RCPP_EXPOSED_AS(NoBounds)
RCPP_EXPOSED_AS(Bounds<double>)
RCPP_EXPOSED_AS(Bounds<arma::mat>)

RCPP_MODULE(OracleModule) {
  declare_bounds<arma::mat>("mat");
  declare_bounds<double>("double");

  Rcpp::class_<NoBounds>("NoBounds")
      .constructor()
      .method("validate", &NoBounds::validate);

  declare_penalty<arma::mat>("mat");
  declare_penalty<double>("double");
}
