#ifndef gL0Learn_H
#define gL0Learn_H
#include "RcppArmadillo.h"
#include "oracle.h"
#include "fitmodel.h"
#include "gap.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <chrono>
#include <thread>

typedef std::vector<std::tuple<arma::uword, arma::uword>> coordinate_vector;

struct CDParams
{
    const double M; 
    const double atol;
    const double rtol;
    const GapMethod gap_method;
    const bool one_normalize;
    const size_t max_iter;
    const double l0;
    const double l1;
    const double l2;
    
    CDParams(const double M,
             const double atol,
             const double rtol,
             const GapMethod gap_method,
             const bool one_normalize,
             const size_t max_iter,
             const double l0,
             const double l1,
             const double l2) : 
        M{M}, atol{atol}, rtol{rtol}, gap_method{gap_method}, 
        one_normalize{one_normalize}, max_iter{max_iter}, l0{l0}, l1{l1}, l2{l2}
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
            this->costs.reserve(this->params.max_iter);
        };
            
        
        void inner_fit();
        const fitmodel fit();
        
        bool psi_row_fit(const arma::uword row_ix);
        const fitmodel fitpsi();
        
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
        std::vector<double> costs;
};

template<class TY, class TR, class TT>
const fitmodel CD<TY, TR, TT>::fit(){
    // Rcpp::Rcout << "fit called \n";
    
    auto old_objective = std::numeric_limits<double>::infinity();
    auto cur_objective = std::numeric_limits<double>::infinity();
    
    std::size_t cur_iter = 0;
    
    while ((cur_iter <= this->params.max_iter) && !this->converged(old_objective, cur_objective, cur_iter)){
        this->inner_fit();
        old_objective = cur_objective;
        cur_objective = this->compute_objective();
        this->costs.push_back(cur_objective);
        cur_iter ++;
        Rcpp::Rcout << "current_iter: " << cur_iter << " cur_objective = " << cur_objective << "\n";
    } 
    return fitmodel(this->theta, this->R, this->costs);
}

template<class TY, class TR, class TT>
void CD<TY, TR, TT>::inner_fit(){
    const size_t p = this->Y.n_cols;
    for (auto ij: this->active_set){
        const auto i = std::get<0>(ij);
        const auto j = std::get<1>(ij);
        
        const auto old_theta_ij = this->theta(i, j);
        const auto old_theta_ji = this->theta(j, i);
        const auto old_theta_ii = this->theta(i, i);
        const auto old_theta_jj = this->theta(j, j);
        
        const auto a = this->S_diag(j)/old_theta_ii + this->S_diag(i)/old_theta_jj;
        const auto b = 2*((arma::dot(this->Y.col(j), this->R.col(i)) - old_theta_ji*this->S_diag(j))/old_theta_ii +
                          (arma::dot(this->Y.col(i), this->R.col(j)) - old_theta_ij*this->S_diag(i))/old_theta_jj);
        const auto new_theta = overleaf_Q_L0L2reg(a, b, this->params.l0, this->params.l2);
        
        this->theta(i, j) = new_theta;
        this->theta(j, i) = new_theta;
        
        this->R.col(i) += (new_theta - old_theta_ij)*this->Y.col(j);
        this->R.col(j) += (new_theta - old_theta_ji)*this->Y.col(i);
    }
        
    for (auto i=0; i < p; i++){
        this->R.col(i) -= this->theta(i, i)*this->Y.col(i);
        this->theta(i, i) = R_nl(this->S_diag(i), arma::dot(this->R.col(i), this->R.col(i)));
        this->R.col(i) += this->theta(i, i)*this->Y.col(i);
    }
}

template<class TY, class TR, class TT>
const fitmodel CD<TY, TR, TT>::fitpsi(){
    // Rcpp::Rcout << "fitpsi called \n";
    static_cast<void>(this->fit());
    const arma::uword p = this->Y.n_cols;
    
    for (auto i=0; i<100; i++){ // TODO: Elevant 100 to Parameter
        // Rcpp::Rcout << "Loop " << i << " \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        bool swap = false;
        
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){
            // Rcpp::Rcout << "Loop " << i << "swap "<< row_ix << " \n";
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            swap = swap || this->psi_row_fit(row_ix);
            if (swap){break;}
        }
        // Rcpp::Rcout << "Loop " << i << " swap " << swap << " \n";
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){
            this->R.col(row_ix) -= this->theta(row_ix, row_ix)*this->Y.col(row_ix);
            this->theta(row_ix, row_ix) = R_nl(this->S_diag(row_ix), arma::dot(this->R.col(row_ix), this->R.col(row_ix)));
            this->R.col(row_ix) += this->theta(row_ix, row_ix)*this->Y.col(row_ix);
        }
        
        if (!swap){
            break;
        } else {
            Rcpp::Rcout << "Loop " << i << " swap cost: " << this->compute_objective() << " \n";
            static_cast<void>(this->fit());
        }
    }
        
    return fitmodel(this->theta, this->R, this->costs);
}

template<class TY, class TR, class TT>
bool CD<TY, TR, TT>::psi_row_fit(const arma::uword row_ix){
    // Rcpp::Rcout << "psi_row_fit \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    arma::mat::const_row_iterator it = this->theta.begin_row(row_ix);
    arma::mat::const_row_iterator end = this->theta.end_row(row_ix);

    std::vector<arma::uword> zero_indices;
    std::vector<arma::uword> non_zero_indices;
    
    for(arma::uword index = 0; it != end; ++it, ++index){
        // zero_indices and non_zero_indices will not form the set (1, ..., p)
        // as if theta(i, i) is non-zero it will not go into non_zero_indicies
        if ((*it) != 0){
            if (index != row_ix){non_zero_indices.push_back(index);}
        } else {
            zero_indices.push_back(index);
        }
    }
    // Rcpp::Rcout << "Zeros found \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // If every item is either 0 or non-zero then swapping is pointless
    if (zero_indices.empty() || non_zero_indices.empty()){return false;}
    
    const arma::uvec zeros(zero_indices);
    const arma::uvec non_zeros(non_zero_indices);
    
    const arma::vec theta_diag = arma::vec(this->theta.diag());

    for(auto j: non_zero_indices){
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        R.col(row_ix) -= this->theta(row_ix, j)*this->Y.col(j);
        R.col(j) -= this->theta(j, row_ix)*this->Y.col(row_ix);
        this->theta(j, row_ix) = 0;
        this->theta(row_ix, j) = 0;

        const double aj = this->S_diag[row_ix]/theta_diag(j) + this->S_diag[j]/theta_diag(row_ix);
        const double bj = 2*((arma::dot(this->Y.col(j), this->R.col(row_ix))/theta_diag(row_ix))
                          +  (arma::dot(this->Y.col(row_ix), this->R.col(j))/theta_diag(j)));

        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        const std::tuple<double, double> theta_f = overleaf_Q_L0L2reg_obj(aj,
                                                                          bj,
                                                                          this->params.l0,
                                                                          this->params.l2);
        const double theta = std::get<0>(theta_f);
        const double f = std::get<1>(theta_f);

        const arma::vec a_vec = (this->S_diag(row_ix)/theta_diag(zeros) 
                               + this->S_diag(zeros)/theta_diag(row_ix));
        const arma::vec b_vec = 2*(((this->Y.cols(zeros).t()*this->R.col(row_ix))/theta_diag(row_ix))
                                 + ((this->R.cols(zeros).t()*this->Y.col(row_ix))/theta_diag(zeros)));
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " theta_diag " << theta_diag << " \n";
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " b_vec " << b_vec << " \n";
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj vec \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        const std::tuple<arma::vec, arma::vec> thetas_fs = overleaf_Q_L0L2reg_obj(a_vec,
                                                                                  b_vec,
                                                                                  this->params.l0,
                                                                                  this->params.l2);
        
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " std::get \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        const arma::vec thetas = std::get<0>(thetas_fs);
        const arma::vec fs = std::get<1>(thetas_fs);
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " fs: " << fs << " \n";
        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " thetas: "<< thetas <<"\n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (f < fs.min()){
            // Rcpp::Rcout << "Non Zero Index Loop: " << j << "f <= fs.min()" << "\n";
            this->theta(row_ix, j) = theta;
            this->theta(j, row_ix) = theta;
            this->R.col(row_ix) += theta*this->Y.col(j);
            this->R.col(j) += theta*this->Y.col(row_ix);
        } else {
            // Rcpp::Rcout << "Non Zero Index Loop: " << j << " Else" << "\n";
            const auto ell = arma::index_min(fs);
            // Rcpp::Rcout << "Non Zero Index Loop: " << j << " ell = " << ell << "\n";
            const auto k = zeros(ell);
            // Rcpp::Rcout << "Non Zero Index Loop: " << j << " k = zeros(ell) = " << k << "\n";
            const auto k_theta = thetas(ell);
            // Rcpp::Rcout << "Non Zero Index Loop: " << j << "k_theta = " << k_theta << "\n";
            this->theta(row_ix, k) = k_theta;
            this->theta(k, row_ix) = k_theta;
            this->R.col(row_ix) += k_theta*this->Y.col(k);
            this->R.col(k) += k_theta*this->Y.col(row_ix);
            return true;
        }
    }
    
    return false;
}


// template<class TY, class TR, class TT>
// bool inline CD<TY, TR, TT>::converged(const double old_objective,
//                                       const double cur_objective,
//                                       const size_t cur_iter){
//     return ((cur_objective <= this->params.atol) 
//                 || ((cur_iter > 1) && (std::abs(old_objective - cur_objective) < old_objective * this->params.rtol)));
// };


template<class TY, class TR, class TT>
bool inline CD<TY, TR, TT>::converged(const double old_objective,
                                      const double cur_objective,
                                      const size_t cur_iter){
    return ((cur_iter > 1) && (relative_gap(old_objective, cur_objective, this->params.gap_method, this->params.one_normalize) <= this->params.rtol));
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
        const bool is_nnz_theta_ij = abs_theta_ij >= 1e-14; // Lift parameter to configuartion
        cost += is_nnz_theta_ij*(this->params.l0 
                                 + abs_theta_ij*this->params.l1 
                                 + theta_ij*theta_ij*this->params.l2);
    }
    
    return cost;
};
    
template class CD<const arma::mat, arma::mat, arma::mat>;

#endif // gL0Learn_H

