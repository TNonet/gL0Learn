#include "oracle.h"

double objective(const arma::mat &theta, const arma::mat &R) {
  /*
   *  Objective = \sum_{i=1}^{p}(||<Y, theta[i, :]>||_2 - log(theta[i, i]))
   *
   *  Notes
   *  -----
   *  If we use a sparse form of TT, the objective can be sped up in the active
   * set calculation.
   */
  auto theta_diag = arma::vec(theta.diag());
  double cost = -arma::sum(arma::log(theta_diag));

  const auto p = R.n_cols;
  // TODO: Rewrite this as sum(SQUARE(R).sum(columns)/theta_diag);
  for (auto i = 0; i < p; i++) {
    cost += arma::dot(R.col(i), R.col(i)) / theta_diag[i];
  }

  return cost;
}
