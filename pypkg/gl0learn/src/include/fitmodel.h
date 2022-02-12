#ifndef H_FITMODEL
#define H_FITMODEL
#include "arma_includes.h"


struct fitmodel
{
    const arma::mat theta;
    const arma::mat R;
    const std::vector<double> costs;
    const std::vector<std::size_t > active_set_size;
    

    fitmodel() = default;
    fitmodel(const fitmodel &f) : 
        theta(std::move(f.theta)), 
        R(std::move(f.R)),
        costs(std::move(f.costs)),
        active_set_size(std::move(f.active_set_size)) {};
    fitmodel(const arma::mat& theta,
             const arma::mat& R,
             const std::vector<double>& costs,
             const std::vector<std::size_t>& active_set_size):
        theta(std::move(theta)),
        R(std::move(R)),
        costs(std::move(costs)),
        active_set_size(std::move(active_set_size)){
        }
};

#endif // H_FITMODEL