#ifndef H_FITMODEL
#define H_FITMODEL
#include "RcppArmadillo.h"
#include <chrono>
#include <thread>


struct fitmodel
{
    const arma::mat theta;
    const arma::mat R;

    fitmodel() = default;
    fitmodel(const fitmodel &) = default;
    fitmodel(const arma::mat theta,
             const arma::mat R):
        theta(std::move(theta)),
        R(std::move(R)) {
        Rcpp::Rcout << "fitmodel constructor\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
};

#endif // H_FITMODEL