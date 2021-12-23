#ifndef H_FITMODEL
#define H_FITMODEL
#include "RcppArmadillo.h"
#include <chrono>
#include <thread>


struct fitmodel
{
    const arma::mat theta;
    const arma::mat R;
    const std::vector<double> costs;

    fitmodel() = default;
    fitmodel(const fitmodel &f) : theta(std::move(f.theta)), R(std::move(f.R)), costs(std::move(f.costs)) {};
    fitmodel(const arma::mat& theta,
             const arma::mat& R,
             const std::vector<double>& costs):
        theta(std::move(theta)),
        R(std::move(R)),
        costs(std::move(costs)){
        }
};

#endif // H_FITMODEL