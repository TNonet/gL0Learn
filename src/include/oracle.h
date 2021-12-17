#ifndef H_ORACLE
#define H_ORACLE

#include <chrono>
#include <thread>
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

template <class T>
T inline R_nl(const T a, const T b){
    return (1 + arma::sqrt(1 + 4 % a % b)) / (2*a);
}

template <>
double inline R_nl(const double a, const double b){
    return (1 + std::sqrt(1 + 4 * a * b)) / (2 * a);
}


// double inline prox_L0L2reg(const double beta, const double l0, const double l2, const double M)
// {
//     const auto inv_two_l2_plus_1 = 1/(l2*2 + 1);
//     const auto val = std::abs(beta)*inv_two_l2_plus_1;
//     
//     if (val <= M){
//         if (val > std::sqrt(2*l0*inv_two_l2_plus_1)){
//             return std::signbit(beta)*val;
//         } else {
//             return 0; 
//         }
//     } else {
//         if (val > 0.5*M + l0/M*inv_two_l2_plus_1){
//             return std::signbit(beta)*M;
//         } else {
//             return 0;
//         }
//     }
// }


// double inline Q_L0L2reg(const double a,
//                         const double b,
//                         const double l0, 
//                         const double l2,
//                         const double M)
// {
//     const auto inv_2a = 1/(2*a);
//     const auto beta = -b * inv_2a;
//     const auto l0_a = l0 * inv_2a;
//     const auto l2_a = l2 * inv_2a;
//     return prox_L0L2reg(beta, l0_a, l2_a, M);
// };
template <class T>
T inline overleaf_prox_L0L2reg(const T beta,
                               const T l0,
                               const T l2)
{
    // Rcpp::Rcout << "overleaf_prox_L0L2reg vec T beta" << beta << " \n";
    // Rcpp::Rcout << "overleaf_prox_L0L2reg vec T l0" <<  l0 << "\n";
    // Rcpp::Rcout << "overleaf_prox_L0L2reg vec T l2" << l2  <<"  \n";
    const T two_l2_plus_1 = 2*l2 + 1;
    const T thresh = arma::sqrt(2*l0%two_l2_plus_1);
    const T max_val = beta/two_l2_plus_1;
    const T abs_beta = arma::abs(beta);
    // Rcpp::Rcout << "abs_gte_thresh vec \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    const auto abs_gte_thresh = abs_beta >= thresh;
    
    // Rcpp::Rcout << "retrurn vec \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return abs_gte_thresh%max_val;
}

template <class T>
T inline hussein_prox_L0L1L2_bounds_reg(const T beta,
                                        const T l0,
                                        const T l1,
                                        const T l2,
                                        const double lb,
                                        const double ub)
{
    // In theroy, we can make lb and ub templated classes as long as we handle arma::clamp properly. Double vs Vector 
    const T two_l2_plus_1 = 2*l2 + 1;
    const T two_l0 = 2*l0;
    const T abs_beta_m_l1 = beta.abs() - l1;
    const T ns_beta_opt_no_l0 = abs_beta_m_l1/two_l2_plus_1;
    const T beta_opt_no_l0 = arma::sign(beta)%ns_beta_opt_no_l0;
    const T beta_opt_bounds = arma::clamp(beta_opt_no_l0, lb, ub);
    const T delta = arma::sqrt(arma::max(arma::pow(abs_beta_m_l1,2) - two_l0%two_l2_plus_1, 0.));
    
    const auto crit_1 = ns_beta_opt_no_l0 >= arma::sqrt(two_l0/two_l2_plus_1);
    const auto crit_2 = (beta_opt_no_l0 - delta <= beta_opt_bounds) & (beta_opt_bounds <= beta_opt_no_l0 + delta);
    
    return beta_opt_bounds*(crit_1 || crit_2);
    
}

template <typename T> 
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
inline T clamp(T x, T low, T high) {
    // -O3 Compiler should remove branches
    if (x < low) 
        x = low;
    if (x > high) 
        x = high;
    return x;
}

template <>
double inline hussein_prox_L0L1L2_bounds_reg(const double beta,
                                             const double l0,
                                             const double l1,
                                             const double l2,
                                             const double lb,
                                             const double ub)
{
    
    const double two_l2_plus_1 = 2*l2 + 1;
    const double two_l0 = 2*l0;
    const double abs_beta_m_l1 = std::fabs(beta) - l1;
    const double ns_beta_opt_no_l0 = abs_beta_m_l1/two_l2_plus_1;
    const double beta_opt_no_l0 = sgn(beta)*ns_beta_opt_no_l0;
    const double beta_opt_bounds = clamp(beta_opt_no_l0, lb, ub);
    const double delta = std::sqrt(std::max(std::pow(abs_beta_m_l1,2) - two_l0*two_l2_plus_1, 0.));
    
    const auto crit_1 = ns_beta_opt_no_l0 >= std::sqrt(two_l0/two_l2_plus_1);
    const auto crit_2 = (beta_opt_no_l0 - delta <= beta_opt_bounds) & (beta_opt_bounds <= beta_opt_no_l0 + delta);
    
    return beta_opt_bounds*(crit_1 || crit_2);
    
}

template <>
double inline overleaf_prox_L0L2reg(const double beta,
                                    const double l0,
                                    const double l2)
{
    const double two_l2_plus_1 = 2*l2 + 1;
    const double thresh = std::sqrt(2*l0*two_l2_plus_1);
    const double max_val = beta/two_l2_plus_1;
    const double abs_beta = std::abs(beta);
        
    if (abs_beta >= thresh){
        return max_val;
    } else {
        return 0;
    }
}

template <class T>
T inline overleaf_Q_L0L2reg(const T a,
                            const T b,
                            const double l0,
                            const double l2)
{
    // Rcpp::Rcout << "overleaf_Q_L0L2reg vec \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const T inv_2a = 1/(2*a);
    const T beta = -b % inv_2a;
    // Rcpp::Rcout << "overleaf_Q_L0L2reg vec T beta " << beta << "\n";
    const T l0_a = l0 * inv_2a;
    const T l2_a = l2 *inv_2a;
    return overleaf_prox_L0L2reg(beta, l0_a, l2_a);
};

template <>
double inline overleaf_Q_L0L2reg(const double a,
                                 const double b,
                                 const double l0,
                                 const double l2)
{
    const double inv_2a = 1/(2*a);
    const double beta = -b * inv_2a;
    const double l0_a = l0 * inv_2a;
    const double l2_a = l2 * inv_2a;
    return overleaf_prox_L0L2reg(beta, l0_a, l2_a);
};

template <class T>
std::tuple<T, T> inline overleaf_Q_L0L2reg_obj(const T a,
                                               const T b,
                                               const double l0,
                                               const double l2)
{
    // Rcpp::Rcout << "overleaf_Q_L0L2reg_obj vec \n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const T x = overleaf_Q_L0L2reg(a, b, l0, l2);
    // Rcpp::Rcout << "overleaf_Q_L0L2reg_obj vec T x" << x << "\n";
    return {x, (a+l2)%x%x + b%x + l0*(x!=0)};
};

template <>
std::tuple<double, double> inline overleaf_Q_L0L2reg_obj(const double a,
                                                         const double b,
                                                         const double l0,
                                                         const double l2)
{
    const double x = overleaf_Q_L0L2reg(a, b, l0, l2);
    return {x, (a+l2)*x*x + b*x + l0*(x!=0)};
};


#endif // H_ORACLE