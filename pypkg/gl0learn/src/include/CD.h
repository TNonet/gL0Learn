#ifndef CD_H
#define CD_H
#include "arma_includes.h"
#include "oracle.h"
#include "fitmodel.h"
#include "gap.h"
#include "active_set.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <chrono>
#include <thread>
#include <iomanip>


template <class O>
struct CDParams
{
    const double atol;
    const double rtol;
    const GapMethod gap_method;
    const bool one_normalize;
    const size_t max_iter;
    const O oracle;
    const std::string algorithm;
    
    CDParams(const double atol,
             const double rtol,
             const GapMethod gap_method,
             const bool one_normalize,
             const size_t max_iter,
             const O& oracle,
             const std::string algorithm) : 
        atol{atol}, rtol{rtol}, gap_method{gap_method}, 
        one_normalize{one_normalize}, max_iter{max_iter}, oracle{oracle},
        algorithm{algorithm} {};
};


template <class TY, class TR, class TT, class TP>
class CD
{
    public: 
        CD(const TY& Y,
           const TT& theta,
           const TP& params,
           const coordinate_vector active_set,
           const coordinate_vector super_active_set): 
            Y{Y},  S_diag{arma::sum(arma::square(Y), 0)}, params{params},
            super_active_set{super_active_set}
        {
            this->theta = TT(theta);
            this->active_set = coordinate_vector(active_set); // Ensure a copy is taken.
            this->R = this->Y*this->theta;
            this->costs.reserve(this->params.max_iter);
            this->active_set_size.reserve(this->params.max_iter);
        };
            
        void restrict_active_set();
        void l0learn_restrict_active_set();
        coordinate_vector active_set_expansion(const coordinate_vector& search_space);
        void inner_fit();
        const fitmodel fit();
        
        bool psi_row_fit(const arma::uword row_ix);
        const fitmodel fitpsi();
        
        coordinate_vector super_active_set_minus_active_set();
        
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
        coordinate_vector super_active_set;
        std::vector<double> costs;
        std::vector<std::size_t> active_set_size;
        coordinate_vector previous_support = coordinate_vector();
        std::size_t same_support_count = 0;
};

template <class TY, class TR, class TT, class TP>
const fitmodel CD<TY, TR, TT, TP>::fit(){
    // Accepts current state for active_set and theta.
    /*
     *  CD Algorithim:
     *  Given:
     *      active_set, AS,: vector of coordinates sorted lexicographically 
     *      super_active_set, SAS: vector of coordinates sorted lexicographically 
     *          such that every item in the active active_set exists in super_active_set.
     *      theta: (p, p) symmetric matrix with non-zero diagonals
     *          such that an item theta[i, j] in SAS but not in AS must be zero
     *          
     *  Steps:
     *      old_objective <- Inf;
     *      cur_objective <- Inf:
     *      
     *      For iter in 0, ..., max_iter-1:
     *          theta[AS] <- updated by Hussein's operator;
     *          old_objective, cur_objective <- compute_objective(), old_objective
     *          
     *          if converged(old_objective, cur_objective):
     *              incr_AS <- items in SAS - AS that want to to be non-zero;
     *              if incr_AS is empty:
     *                  AS <- support of theta
     *                  DONE (RETURN AS, theta)
     *              else:
     *                  AS <- AS U incr_AS
     */
    auto old_objective = std::numeric_limits<double>::infinity();
    auto cur_objective = std::numeric_limits<double>::infinity();
    
    std::size_t cur_iter = 0;
    
    // RUN CD on AS until convergence
    while (cur_iter <= this->params.max_iter){
        this->inner_fit(); // Fits on active_set
        old_objective = cur_objective;
        cur_objective = this->compute_objective();
        this->costs.push_back(cur_objective);
        this->active_set_size.push_back(this->active_set.size());
        cur_iter ++;
        COUT << "current_iter: " << cur_iter << " cur_objective = " << cur_objective << "\n";
        
        if (this->converged(old_objective, cur_objective, cur_iter)){
            const coordinate_vector values_to_check = sorted_vector_difference(this->super_active_set, this->active_set);
            
            if (values_to_check.empty()){ break; } 
            
            coordinate_vector add_to_active_set = this->active_set_expansion(values_to_check);
            
            if (add_to_active_set.empty()){ break; }
            
            this->active_set = insert_sorted_vector_into_sorted_vector(
                this->active_set, add_to_active_set);
        }
        
    } 
    this->restrict_active_set();
    return fitmodel(this->theta, this->R, this->costs, this->active_set_size);
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
void CD<TY, TR, TT, TP>::l0learn_restrict_active_set(){
    
    coordinate_vector new_support = support_from_active_set(this->theta, this->active_set);
    if (new_support == this->previous_support){
        this->same_support_count++;
        
        if (this->same_support_count == 5){ // TODO: Elevate parameter! 
            this->active_set = new_support;
        }
    } else {
        this->same_support_count = 0;
    }
    this->previous_support = new_support;   
}


template <class TY, class TR, class TT, class TP>
coordinate_vector CD<TY, TR, TT, TP>::active_set_expansion(const coordinate_vector& search_space){
    
    const size_t p = this->Y.n_cols;
    const arma::vec theta_diag = arma::vec(this->theta.diag());
    
    const arma::mat ytr = this->Y.t()*this->R;
    
    coordinate_vector items_to_expand_active_set_by;
    items_to_expand_active_set_by.reserve(p);
    for (auto const &ij: search_space){
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        const double a = this->S_diag(j)/theta_diag(i) + this->S_diag(i) / theta_diag(j);
        const double b = 2*ytr(j, i)/theta_diag(i) + 2*ytr(i, j)/theta_diag(j);
        
        if (this->params.oracle.Q(a, b, i, j) != 0){
            items_to_expand_active_set_by.push_back(ij);
        }
        
    }
    return items_to_expand_active_set_by;
}



template <class TY, class TR, class TT, class TP>
void CD<TY, TR, TT, TP>::inner_fit(){
    const size_t p = this->Y.n_cols;
    
    const arma::uvec random_order = arma::randperm(this->active_set.size());
    
    arma::uvec::const_iterator it = random_order.begin();
    const arma::uvec::const_iterator it_end = random_order.end();
    
    for (; it != it_end; ++it){
        auto const ij = this->active_set[*it];
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        
        const double old_theta_ij = this->theta(i, j);
        const double old_theta_ji = this->theta(j, i);
        const double old_theta_ii = this->theta(i, i);
        const double old_theta_jj = this->theta(j, j);
        
        const double a = this->S_diag(j)/old_theta_ii + this->S_diag(i)/old_theta_jj;
        const double b = 2*((arma::dot(this->Y.col(j), this->R.col(i)) - old_theta_ji*this->S_diag(j))/old_theta_ii +
                          (arma::dot(this->Y.col(i), this->R.col(j)) - old_theta_ij*this->S_diag(i))/old_theta_jj);
        const double new_theta = this->params.oracle.Q(a, b, i, j);
        
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
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    COUT << "fitpsi called \n";
    static_cast<void>(this->fit());
    COUT << "Pre psi cost: " << this->compute_objective() << " \n";
    const arma::uword p = this->Y.n_cols;
    
    for (auto i=0; i<100; i++){ // TODO: Elevant 100 to Parameter
        COUT << "PSI iter: " << i << " \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bool swap = false;
        
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){ // TODO: Shuffle the order at which we look rows
            COUT << "PSI iter: " << i << " Swapping row: " << row_ix << "\n";
            swap = swap || this->psi_row_fit(row_ix); // TODO: This needs to properly update the active set.
            if (swap){break;}
        }
        
        for (arma::uword row_ix=0; row_ix < p; row_ix ++){
            // TODO: Do we need to update every diagonal?
            this->R.col(row_ix) -= this->theta(row_ix, row_ix)*this->Y.col(row_ix);
            this->theta(row_ix, row_ix) = R_nl(this->S_diag(row_ix), arma::dot(this->R.col(row_ix), this->R.col(row_ix)));
            this->R.col(row_ix) += this->theta(row_ix, row_ix)*this->Y.col(row_ix);
        }
        
        if (!swap){
            break;
        } else {
            COUT << "PSI iter: " << i << " Post Swap cost: " << this->compute_objective() << " \n";
            static_cast<void>(this->fit());
            COUT << "PSI iter: " << i << " Post Swap Fit cost: " << this->compute_objective() << " \n";
        }
    }
        
    return fitmodel(this->theta, this->R, this->costs, this->active_set_size);
}

template <class TY, class TR, class TT, class TP>
bool CD<TY, TR, TT, TP>::psi_row_fit(const arma::uword row_ix){
    COUT << "psi_row_fit \n";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

    for(auto const &j: non_zero_indices){
        // COUT << "Non Zero Index Loop: (" << row_ix << ", " << j << ") \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        R.col(row_ix) -= this->theta(row_ix, j)*this->Y.col(j);
        R.col(j) -= this->theta(j, row_ix)*this->Y.col(row_ix);
        this->theta(j, row_ix) = 0;
        this->theta(row_ix, j) = 0;

        const double aj = this->S_diag[row_ix]/theta_diag(j) + this->S_diag[j]/theta_diag(row_ix);
        const double bj = 2*((arma::dot(this->Y.col(j), this->R.col(row_ix))/theta_diag(row_ix))
                          +  (arma::dot(this->Y.col(row_ix), this->R.col(j))/theta_diag(j)));

        // COUT << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj \n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        //const std::tuple<double, double> theta_f = this->params.oracle.Qobj(aj, bj);
        const std::tuple<double, double> theta_f = this->params.oracle.Qobj(aj, bj, row_ix, j);
        
        const double theta = std::get<0>(theta_f);
        const double f = std::get<1>(theta_f);

        const arma::vec a_vec = (this->S_diag(row_ix)/theta_diag(zeros) 
                               + this->S_diag(zeros)/theta_diag(row_ix));
        const arma::vec b_vec = 2*(((this->Y.cols(zeros).t()*this->R.col(row_ix))/theta_diag(row_ix))
                                 + ((this->R.cols(zeros).t()*this->Y.col(row_ix))/theta_diag(zeros)));
        // COUT << "Non Zero Index Loop: " << j << " theta_diag " << theta_diag << " \n";
        // COUT << "Non Zero Index Loop: " << j << " b_vec " << b_vec << " \n";
        // COUT << "Non Zero Index Loop: " << j << " overleaf_Q_L0L2reg_obj vec \n";
        
        // COUT << "params.oracle.Qobj(a_vec, b_vec, row_ix) \n";
        // COUT << "params.oracle.Qobj(" << a_vec.size() << ", " <<  b_vec.size() << ", " << row_ix << ")\n";
        // COUT << "zeros.size() = " << zeros.size() <<"\n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //const std::tuple<arma::vec, arma::vec> thetas_fs = this->params.oracle.Qobj(a_vec, b_vec);
        
        const std::tuple<arma::vec, arma::vec> thetas_fs = this->params.oracle.Qobj(a_vec, b_vec, row_ix, zeros);
        const arma::vec thetas = std::get<0>(thetas_fs);
        const arma::vec fs = std::get<1>(thetas_fs);
        
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // COUT << "After this = " << zeros.size() <<"\n";
        // COUT << "Non Zero Index Loop: " << j << " fs: " << fs << " \n";
        // COUT << "Non Zero Index Loop: " << j << " thetas: "<< thetas <<"\n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (f < fs.min()){
            // COUT << "Non Zero Index Loop: " << j << "f <= fs.min()" << "\n";
            this->theta(row_ix, j) = theta;
            this->theta(j, row_ix) = theta;
            this->R.col(row_ix) += theta*this->Y.col(j);
            this->R.col(j) += theta*this->Y.col(row_ix);
            COUT << "No swap for (" << row_ix << ", " << j << ") \n";
        } else {
            /* If a swap is accepted, (`k` <- NNZ for `j` <- 0) in row `row_ix`,
             * We will need to updated AS to ensure it contains `k`. Lucikly,
             * We can keep `j` in the AS. 
             * Since AS is a sorted a coordinate vector such that for each pair 
             * of elements (p1, p2) in AS p1 will be found before p2 if and only
             * if p1 < p2. That is a lexicographical comparison between the 2D
             * coordinates p1 and p2.
             * Therefore we search AS for the first item, u, that is larger than 
             * (row_ix, k), using `lower_bound`. We then insert (row_ix, k)
             *  before u.
            */
            // COUT << "Non Zero Index Loop: " << j << " Else" << "\n";
            const auto ell = arma::index_min(fs);
            // COUT << "Non Zero Index Loop: " << j << " ell = " << ell << "\n";
            const auto k = zeros(ell);
            // COUT << "Non Zero Index Loop: " << j << " k = zeros(ell) = " << k << "\n";
            const auto k_theta = thetas(ell);
            // COUT << "Non Zero Index Loop: " << j << "k_theta = " << k_theta << "\n";
            this->theta(row_ix, k) = k_theta;
            this->theta(k, row_ix) = k_theta;
            this->R.col(row_ix) += k_theta*this->Y.col(k);
            this->R.col(k) += k_theta*this->Y.col(row_ix);
            
            // Find location to insert (row_ix, k)
            const coordinate row_ix_k = {row_ix, k};
            coordinate_vector::iterator low = std::lower_bound(this->active_set.begin(),
                                                               this->active_set.end(),
                                                               row_ix_k);
            
            const size_t loc = low - this->active_set.begin();
            
            // START: Debug Printing statements
            const auto active_set_size = this->active_set.size();
            
            COUT << "Swap for (" << row_ix << ", " << j << ") with (" << row_ix << ", " << k << ")\n";
            COUT << "AS before insert:\n";
            COUT << "AS size = " << active_set_size << " \n";
            COUT << " AS(loc-1) = AS(" << loc - 1 << ")" << this->active_set.at(loc -1 ) << "\n";
            
            if (loc >= active_set_size){
                COUT << " AS has no loc element\n";
            } else{
                COUT << " AS(loc)= " << this->active_set.at(loc) << "\n";
                if (loc + 1 >= active_set_size){
                    COUT << " AS has no loc+1 element\n";
                } else {
                    COUT << " AS(loc+1)= " << this->active_set.at(loc + 1) << "\n";
                }
            }
            // END: Debug Printing statements
            
            if (this->active_set.at(loc - 1) != row_ix_k){
                this->active_set.insert(low, row_ix_k);
            }
            
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
    
    for (auto const &ij: this->active_set){
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        const double theta_ij = this->theta(i, j);
        cost += this->params.oracle.penalty.cost(theta_ij, i, j);
    }
    
    return cost;
};
    
#endif // CD_H

