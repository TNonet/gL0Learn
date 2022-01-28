#ifndef GL0LEARN_H
#define GL0LEARN_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "oracle.h"
#include "fitmodel.h"


// template <class T, template <class> class P, class E, class B> 
// const fitmodel gL0LearnFit(const T& Y,
//                            const T& theta_init,
//                            CDParams<B, P, E>& params){
//     
//     // TODO, pass in active set
//     coordinate_vector active_set = {};
//     const auto p = Y.n_cols;
//     for (arma::uword i=0; i<p; i++){
//         for (arma::uword j=i+1; j<p; j++){
//             active_set.push_back({i, j});
//         }
//     }
//     
//     CD<const T, T, T, CDParams<B, P, E>> x = CD<const T, T, T, CDParams<B, P, E>>(Y, theta_init, params, active_set);
// // 
//     if (params.algorithm == "CD"){
//         return x.fit();
//     } else if (params.algorithm == "CDPSI"){
//         return x.fitpsi();
//     } else {
//         Rcpp::stop("Canno't determine algorithm choice");
//     }
// 
// }
// 
// template <class T, class B, class P>
// const fitmodel gL0LearnFit(const T& Y,
//                            const T& theta_init,
//                            const P& l0,
//                            const P& l2,
//                            const B& lows,
//                            const B& highs,
//                            const std::string algorithm,
//                            const double atol,
//                            const double rtol,
//                            const size_t max_iter){
//     
//     const PenaltyL0L2<P> p(l0, l2);
//     const Bounds<B> b(lows, highs);
//     return gL0LearnFit(Y, theta_init, p, b, algorithm, atol, rtol, max_iter);
// }
// 
// template <class T, class B, class P>
// const fitmodel gL0LearnFit(const T& Y,
//                            const T& theta_init,
//                            const P& l0,
//                            const P& l1,
//                            const P& l2,
//                            const B& lows,
//                            const B& highs,
//                            const std::string algorithm,
//                            const double atol,
//                            const double rtol,
//                            const size_t max_iter){
//     
//     const PenaltyL0L1L2<P> p(l0, l1, l2);
//     const Bounds<B> b(lows, highs);
//     return gL0LearnFit(Y, theta_init, p, b, algorithm, atol, rtol, max_iter);
// }
// 
// template <class T, class B, class P>
// const fitmodel gL0LearnFit(const T& Y,
//                            const T& theta_init,
//                            const P& l0,
//                            const P& l2,
//                            const std::string algorithm,
//                            const double atol,
//                            const double rtol,
//                            const size_t max_iter){
//     
//     const PenaltyL0L2<P> p(l0, l2);
//     return gL0LearnFit(Y, theta_init, p, NoBounds(), algorithm, atol, rtol, max_iter);
// }
// 
// template <class T, class B, class P>
// const fitmodel gL0LearnFit(const T& Y,
//                            const T& theta_init,
//                            const P& l0,
//                            const P& l1,
//                            const P& l2,
//                            const std::string algorithm,
//                            const double atol,
//                            const double rtol,
//                            const size_t max_iter){
//     
//     const PenaltyL0L1L2<P> p(l0, l1, l2);
//     return gL0LearnFit(Y, theta_init, p, NoBounds(), algorithm, atol, rtol, max_iter);
// }


template <class T, class O> 
const fitmodel gL0LearnFit(const T& Y,
                           const T& theta_init,
                           const O& oracle,
                           const std::string algorithm,
                           const arma::umat& initial_active_set,
                           const arma::umat& super_active_set, 
                           const double atol,
                           const double rtol,
                           const size_t max_iter){
    
    const auto params = CDParams<O>(atol,
                                    rtol,
                                    GapMethod::both,
                                    true,
                                    max_iter,
                                    oracle,
                                    algorithm);
    
    const coordinate_vector initial_active_set_vec = coordinate_vector_from_matrix(initial_active_set);
    const coordinate_vector super_active_set_vec = coordinate_vector_from_matrix(super_active_set); 
    
    auto cd = CD<const T, T, T, CDParams<O>>(Y, theta_init, params, initial_active_set_vec, super_active_set_vec);
    
    if (algorithm == "CD"){
        Rcpp::Rcout << "FIT CD";
        return cd.fit();
    } else if (algorithm == "CDPSI"){
        Rcpp::Rcout << "FIT CDPSI";
        return cd.fitpsi();
    } else {
        Rcpp::stop("Canno't determine algorithm choice");
    }
}


#endif // GL0LEARN_H