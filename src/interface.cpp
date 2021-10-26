#include "RcppArmadillo.h"
#include "gL0Learn.h"

// [[Rcpp::export]]
Rcpp::List gL0Learn_fit(const arma::mat& Y,
                        arma::mat& theta_init,
                        const double atol,
                        const double rtol,
                        const double M,
                        const double l0, 
                        const double l1, 
                        const double l2,
                        const size_t max_iter) {
    Rcpp::Rcout << "CD test_fit  Start \n";
    //coordinate_vector active_set = {{1, 0}, {2, 0}, {2, 1}};
    
    coordinate_vector active_set = {};
    const auto p = Y.n_cols;
    for (arma::uword i=0; i<p; i++){
        for (arma::uword j=i+1; j<p; j++){
            active_set.push_back({i, j});
        }
    }
    
    const CDParams params = CDParams(M, atol, rtol, max_iter, l0, l1, l2);
    
    CD<arma::mat, arma::mat, arma::mat> x = CD<arma::mat, arma::mat, arma::mat>(Y, theta_init, params, active_set);
    const fitmodel l = x.fit();
    
    Rcpp::Rcout << "fitmodel finished\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    return(Rcpp::List::create(Rcpp::Named("theta") = l.theta,
                              Rcpp::Named("R") = l.R));
}
