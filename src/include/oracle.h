#ifndef H_ORACLE
#define H_ORACLE

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

double inline R_nl(const double a, const double b){
    return (1 + std::sqrt(1 + 4 * a * b)) / (2 * a);
}


double inline prox_L0L2reg(const double beta, const double l0, const double l2, const double M)
{
    const auto inv_two_l2_plus_1 = 1/(l2*2 + 1);
    const auto val = std::abs(beta)*inv_two_l2_plus_1;
    
    if (val <= M){
        if (val > std::sqrt(2*l0*inv_two_l2_plus_1)){
            return std::signbit(beta)*val;
        } else {
            return 0; 
        }
    } else {
        if (val > 0.5*M + l0/M*inv_two_l2_plus_1){
            return std::signbit(beta)*M;
        } else {
            return 0;
        }
    }
}


double inline Q_L0L2reg(const double a,
                        const double b,
                        const double l0, 
                        const double l2,
                        const double M)
{
    const auto inv_2a = 1/(2*a);
    const auto beta = -b * inv_2a;
    const auto l0_a = l0 * inv_2a;
    const auto l2_a = l2 * inv_2a;
    return prox_L0L2reg(beta, l0_a, l2_a, M);
};


double inline overleaf_prox_L0L2reg(const double beta,
                                    const double l0,
                                    const double l2)
{
    const auto two_l2_plus_1 = 2*l2 + 1;
    const auto thresh = std::sqrt(2*l0*two_l2_plus_1);
    const auto max_val = beta/two_l2_plus_1;
    const auto abs_beta = std::abs(beta);
    
    
    if (abs_beta > thresh){
        return max_val;
    } else if (abs_beta < thresh) {
        return 0;
    } else {
        return max_val/2;
    }
}


double inline overleaf_Q_L0L2reg(const double a,
                                 const double b,
                                 const double l0,
                                 const double l2)
{
    const auto inv_2a = 1/(2*a);
    const auto beta = -b * inv_2a;
    const auto l0_a = l0 * inv_2a;
    const auto l2_a = l2 * inv_2a;
    return overleaf_prox_L0L2reg(beta, l0_a, l2_a);
};


#endif // H_ORACLE