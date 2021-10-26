#ifndef gL0Learn_H
#define gL0Learn_H
#include "RcppArmadillo.h"
#include "oracle.h"
#include "fitmodel.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <chrono>
#include <thread>

typedef std::vector<std::tuple<arma::uword, arma::uword>> coordinate_vector;

struct CDParams
{
    const double M; 
    const double atol;
    const double rtol;
    const size_t max_iter;
    const double l0;
    const double l1;
    const double l2;
    
    CDParams(const double M,
             const double atol,
             const double rtol,
             const size_t max_iter,
             const double l0,
             const double l1,
             const double l2) : 
        M{M}, atol{atol}, rtol{rtol}, max_iter{max_iter}, l0{l0}, l1{l1}, l2{l2}
    {
    };
};

template <class TY, class TR, class TT>
class CD
{
    public: 
        CD(const TY& Y,
           TT& theta,
           const CDParams params,
           coordinate_vector active_set): 
            Y{Y},  S_diag{arma::sum(arma::square(Y), 0)}, params{params}
        {
            this->theta = theta;
            this->active_set = active_set;
            this->R = this->Y*this->theta;
        };
            
        
        void inner_fit();
        const fitmodel fit();
        
        double inline compute_objective();
        bool inline converged(const double old_objective,
                              const double cur_objective,
                              const size_t cur_iter);
        
    private:
        const TY Y;
        TR R;
        TT theta;
        const arma::rowvec S_diag;
        const CDParams params;
        coordinate_vector active_set;
};

template<class TY, class TR, class TT>
const fitmodel CD<TY, TR, TT>::fit(){
    Rcpp::Rcout << "fit called \n";
    
    auto old_objective = std::numeric_limits<double>::infinity();
    auto cur_objective = std::numeric_limits<double>::infinity();
    
    std::size_t cur_iter = 0;
    
    while ((cur_iter <= this->params.max_iter) && !this->converged(old_objective, cur_objective, cur_iter)){
        this->inner_fit();
        old_objective = cur_objective;
        cur_objective = this->compute_objective();
        cur_iter ++;
        Rcpp::Rcout << "current_iter: " << cur_iter << " cur_objective = " << cur_objective << "\n";
    } 
    Rcpp::Rcout << "Exporitng fit\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return fitmodel(this->theta, this->R);
}

template<class TY, class TR, class TT>
void CD<TY, TR, TT>::inner_fit(){
    const size_t p = this->Y.n_cols;
    Rcpp::Rcout << "active set start \n";
    for (auto ij: this->active_set){
        const auto i = std::get<0>(ij);
        const auto j = std::get<1>(ij);
        
        const auto old_theta_ij = this->theta(i, j);
        const auto old_theta_ji = this->theta(j, i);
        const auto old_theta_ii = this->theta(i, i);
        const auto old_theta_jj = this->theta(j, j);
        
        const auto a = this->S_diag(j)/old_theta_ii + this->S_diag(i)/old_theta_jj;
        const auto b = ((2*arma::dot(this->Y.col(j), this->R.col(i)) - 2*old_theta_ij*this->S_diag(j))/old_theta_ii +
                        (2*arma::dot(this->Y.col(i), this->R.col(j)) - 2*old_theta_ji*this->S_diag(i))/old_theta_jj);
        const auto new_theta = Q_L0L2reg(a, b, this->params.l0, this->params.l2, this->params.M);
        
        this->theta(i, j) = new_theta;
        this->theta(j, i) = new_theta;
        
        this->R.col(i) += (new_theta - old_theta_ij)*this->Y.col(j);
        this->R.col(j) += (new_theta - old_theta_ji)*this->Y.col(i);
    }
    Rcpp::Rcout << "active set end \n";
        
    for (auto i=0; i < p; i++){
        this->R.col(i) -= this->theta(i, i)*this->Y.col(i);
        this->theta(i, i) = R_nl(this->S_diag(i), arma::dot(this->R.col(i), this->R.col(i)));
        this->R.col(i) += this->theta(i, i)*this->Y.col(i);
    }
}

template<class TY, class TR, class TT>
bool inline CD<TY, TR, TT>::converged(const double old_objective,
                                      const double cur_objective,
                                      const size_t cur_iter){
    return ((cur_objective <= this->params.atol) 
                || ((cur_iter > 1) && (std::abs(old_objective - cur_objective) < old_objective * this->params.rtol)));
};


template<class TY, class TR, class TT>
double inline CD<TY, TR, TT>::compute_objective(){
    /*
     *  Objective = \sum_{i=1}^{p}(||<Y, theta[i, :]>||_2 - log(theta[i, i]))
     *  
     *  Notes
     *  -----
     *  If we use a sparse form of TT, the objective can be sped up in the active set calculation.
     */
    auto theta_diag = arma::vec(this->theta.diag());
    double cost = - arma::sum(arma::log(theta_diag));
    
    for (auto i=0; i < this->R.n_cols; i++){
        cost += arma::dot(this->R.col(i), this->R.col(i))/theta_diag[i];
    }
    
    for (auto ij: this->active_set){
        const auto theta_ij = this->theta[std::get<0>(ij), std::get<1>(ij)];
        const auto abs_theta_ij = std::abs(theta_ij);
        const bool is_nnz_theta_ij = abs_theta_ij >= 1e-14;
        cost += is_nnz_theta_ij*(this->params.l0 
                                     + abs_theta_ij*this->params.l1 
                                     + theta_ij*theta_ij*this->params.l2);
    }
    
    return cost;
};
    
template class CD<const arma::mat, arma::mat, arma::mat>;

#endif // gL0Learn_H

