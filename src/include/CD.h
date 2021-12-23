#ifndef gL0Learn_H
#define gL0Learn_H
#include "RcppArmadillo.h"
#include "oracle.h"
#include "fitmodel.h"
#include "gap.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <chrono>
#include <thread>


template <class B, template<class> class P, class E>
struct CDParams
{
    const double atol;
    const double rtol;
    const GapMethod gap_method;
    const bool one_normalize;
    const size_t max_iter;
    const P<E> penalty;
    const B bounds;
    const std::string algorithm;
    
    CDParams(const double atol,
             const double rtol,
             const GapMethod gap_method,
             const bool one_normalize,
             const size_t max_iter,
             const P<E> penalty,
             const B bounds,
             const std::string algorithm) : 
        atol{atol}, rtol{rtol}, gap_method{gap_method}, 
        one_normalize{one_normalize}, max_iter{max_iter}, penalty{penalty},
        bounds{bounds}, algorithm{algorithm} {};
    
    auto get_scalar_oracle(const arma::uword row_index,
                           const arma::uword col_index) const{
        return _get_scalar_oracle(this->penalty,
                                  this->bounds,
                                  row_index,
                                  col_index);
    };
    
    auto get_row_oracle(const arma::uword row_index,
                        const arma::uvec col_indicies) const{
        return _get_row_oracle(this->penalty,
                               this->bounds,
                               row_index,
                               col_indicies);
    };
};


template <class TY, class TR, class TT, class TP>
class CD
{
    public: 
        CD(const TY& Y,
           const TT& theta,
           const TP& params,
           coordinate_vector active_set): 
            Y{Y},  S_diag{arma::sum(arma::square(Y), 0)}, params{params}
        {
            this->theta = TT(theta);
            this->active_set = active_set;
            this->R = this->Y*this->theta;
            this->costs.reserve(this->params.max_iter);
        };
            
        void restrict_active_set();
        void expand_active_set();
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
        const TP params;
        coordinate_vector active_set;
        std::vector<double> costs;
};

template <class TY, class TR, class TT, class TP>
const fitmodel CD<TY, TR, TT, TP>::fit(){
    auto old_objective = std::numeric_limits<double>::infinity();
    auto cur_objective = std::numeric_limits<double>::infinity();
    
    std::size_t cur_iter = 0;
    
    while ((cur_iter <= this->params.max_iter) && !this->converged(old_objective, cur_objective, cur_iter)){
        this->inner_fit();
        this->restrict_active_set();
        this->expand_active_set();
        old_objective = cur_objective;
        cur_objective = this->compute_objective();
        this->costs.push_back(cur_objective);
        cur_iter ++;
        Rcpp::Rcout << "current_iter: " << cur_iter << " cur_objective = " << cur_objective << "\n";
    } 
    return fitmodel(this->theta, this->R, this->costs);
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::restrict_active_set(){
    coordinate_vector restricted_active_set;
    // copy only coordinates with non_zero thetas:
    std::copy_if (this->active_set.begin(), 
                  this->active_set.end(), 
                  std::back_inserter(restricted_active_set),
                  [this](const coordinate ij){return (std::get<0>(ij) == std::get<1>(ij)) || (this->theta(std::get<0>(ij), std::get<1>(ij)) != 0);} );
    this->active_set = restricted_active_set;
}

template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::expand_active_set(){
    
    const size_t p = this->Y.n_cols;
    const arma::vec theta_diag = arma::vec(this->theta.diag());
    
    const arma::mat ytr = this->Y.t()*this->R;
    
    // Rcpp::Rcout << "ytr.size" << arma::size(ytr) << "\n";
    
    coordinate first_item = {0, 0};
    
    coordinate_vector expanded_active_set;
    for (auto ij: this->active_set){
        while (first_item < ij){
            const double i0 = std::get<0>(first_item);
            const double j0 = std::get<1>(first_item);
            
            if (i0 != j0){
                // Rcpp::Rcout << "ytr("<< i0 << ", " << j0 << ") \n";
                
                const double a = this->S_diag(j0)/theta_diag(i0) + this->S_diag(i0) / theta_diag(j0);
                const double b = 2*ytr(j0, i0)/theta_diag(i0) + 2*ytr(i0, j0)/theta_diag(j0);
                if (this->params.get_scalar_oracle(i0, j0).Q(a, b) != 0){
                    expanded_active_set.push_back(first_item);
                }
            }
            first_item = inc(first_item, p);
        }
        expanded_active_set.push_back(ij);
        first_item = inc(first_item, p);

    }
    this->active_set = expanded_active_set;
}



template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::inner_fit(){
    const size_t p = this->Y.n_cols;
    for (auto ij: this->active_set){
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        
        const double old_theta_ij = this->theta(i, j);
        const double old_theta_ji = this->theta(j, i);
        const double old_theta_ii = this->theta(i, i);
        const double old_theta_jj = this->theta(j, j);
        
        const double a = this->S_diag(j)/old_theta_ii + this->S_diag(i)/old_theta_jj;
        const double b = 2*((arma::dot(this->Y.col(j), this->R.col(i)) - old_theta_ji*this->S_diag(j))/old_theta_ii +
                          (arma::dot(this->Y.col(i), this->R.col(j)) - old_theta_ij*this->S_diag(i))/old_theta_jj);
        const double new_theta = this->params.get_scalar_oracle(i, j).Q(a, b);
        
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

template <class TY, class TR, class TT, class TP>
const fitmodel CD<TY, TR, TT, TP>::fitpsi(){
    Rcpp::Rcout << "fitpsi called \n";
    static_cast<void>(this->fit());
    const arma::uword p = this->Y.n_cols;
    
    for (auto i=0; i<100; i++){ // TODO: Elevant 100 to Parameter
        Rcpp::Rcout << "PSI iter: " << i << " \n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        bool swap = false;
        
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){
            Rcpp::Rcout << "PSI iter: " << i << " Swapping row: " << row_ix << "\n";
            swap = swap || this->psi_row_fit(row_ix);
            if (swap){break;}
        }
        
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){
            this->R.col(row_ix) -= this->theta(row_ix, row_ix)*this->Y.col(row_ix);
            this->theta(row_ix, row_ix) = R_nl(this->S_diag(row_ix), arma::dot(this->R.col(row_ix), this->R.col(row_ix)));
            this->R.col(row_ix) += this->theta(row_ix, row_ix)*this->Y.col(row_ix);
        }
        
        if (!swap){
            break;
        } else {
            Rcpp::Rcout << "PSI iter: " << i << " Post Swap cost: " << this->compute_objective() << " \n";
            static_cast<void>(this->fit());
            Rcpp::Rcout << "PSI iter: " << i << " Post Swap Fit cost: " << this->compute_objective() << " \n";
        }
    }
        
    return fitmodel(this->theta, this->R, this->costs);
}

template <class TY, class TR, class TT, class TP>
bool CD<TY, TR, TT, TP>::psi_row_fit(const arma::uword row_ix){
    Rcpp::Rcout << "psi_row_fit \n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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
    // If every item is either 0 or non-zero then swapping is pointless
    if (zero_indices.empty() || non_zero_indices.empty()){return false;}
    
    const arma::uvec zeros(zero_indices);
    const arma::uvec non_zeros(non_zero_indices);
    
    const arma::vec theta_diag = arma::vec(this->theta.diag());

    for(auto j: non_zero_indices){
        Rcpp::Rcout << "Non Zero Index Loop: (" << row_ix << ", " << j << ") \n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        R.col(row_ix) -= this->theta(row_ix, j)*this->Y.col(j);
        R.col(j) -= this->theta(j, row_ix)*this->Y.col(row_ix);
        this->theta(j, row_ix) = 0;
        this->theta(row_ix, j) = 0;

        const double aj = this->S_diag[row_ix]/theta_diag(j) + this->S_diag[j]/theta_diag(row_ix);
        const double bj = 2*((arma::dot(this->Y.col(j), this->R.col(row_ix))/theta_diag(row_ix))
                          +  (arma::dot(this->Y.col(row_ix), this->R.col(j))/theta_diag(j)));

        // Rcpp::Rcout << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        //const std::tuple<double, double> theta_f = this->params.oracle.Qobj(aj, bj);
        const std::tuple<double, double> theta_f = this->params.get_scalar_oracle(row_ix, j).Qobj(aj, bj);
        
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
        //const std::tuple<arma::vec, arma::vec> thetas_fs = this->params.oracle.Qobj(a_vec, b_vec);
        const std::tuple<arma::vec, arma::vec> thetas_fs = this->params.get_row_oracle(row_ix, zeros).Qobj(a_vec, b_vec);
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
            Rcpp::Rcout << "No swap for (" << row_ix << ", " << j << ") \n";
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
            Rcpp::Rcout << "Swap for (" << row_ix << ", " << j << ") with (" << row_ix << ", " << k << ")\n";
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


template <class TY, class TR, class TT, class TP>
bool inline CD<TY, TR, TT, TP>::converged(const double old_objective,
                                      const double cur_objective,
                                      const size_t cur_iter){
    return ((cur_iter > 1) && (relative_gap(old_objective, cur_objective, this->params.gap_method, this->params.one_normalize) <= this->params.rtol));
};


template <class TY, class TR, class TT, class TP>
double inline CD<TY, TR, TT, TP>::compute_objective(){
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
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        const double theta_ij = this->theta(i, j);
        cost += this->params.get_scalar_oracle(i, j).penalty.cost(theta_ij);
    }
    
    return cost;
};
    
#endif // gL0Learn_H

