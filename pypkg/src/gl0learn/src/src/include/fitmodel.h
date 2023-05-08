#ifndef H_FITMODEL
#define H_FITMODEL
#include <vector>

#include "arma_includes.h"

struct fitmodel {
  const arma::mat theta;
  const arma::mat R;
  const std::vector<double> costs;
  const std::vector<std::size_t> active_set_size;

  fitmodel(const fitmodel &f)
      : theta(f.theta),
        R(f.R),
        costs(f.costs),
        active_set_size(f.active_set_size) {}
  fitmodel(const arma::mat &theta, const arma::mat &R,
           const std::vector<double> &costs,
           const std::vector<std::size_t> &active_set_size)
      : theta(theta), R(R), costs(costs), active_set_size(active_set_size) {}
};

#endif  // H_FITMODEL
