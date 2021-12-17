#ifndef ORACLE2_H
#define ORACLE2_H
#include "RcppArmadillo.h"
#include "utils.h"


template <class T>
T inline R_nl(const T a, const T b){
    return (1 + arma::sqrt(1 + 4 % a % b)) / (2*a);
}

template <>
double inline R_nl(const double a, const double b){
    return (1 + std::sqrt(1 + 4 * a * b)) / (2 * a);
}


struct NoBounds {
    NoBounds() {};
    
    NoBounds(const NoBounds,
             const arma::uword row_index,
             const arma::uword col_index) {};
    
    NoBounds(const NoBounds,
             const arma::uword row_index,
             const arma::uvec col_indicies) {};
};


template <class T>
struct Bounds {
    const T lows;
    const T highs;
    typedef T type;
    
    Bounds(const T lows, const T highs) : lows{lows}, highs{highs} {};
    
    template <class TT>
    Bounds(const Bounds<TT>,
           const arma::uword row_index,
           const arma::uword col_index);
    
    template <class TT>
    Bounds(const Bounds<TT>,
           const arma::uword row_index,
           const arma::uvec col_indicies);
};

template <>
template <>
inline Bounds<double>::Bounds(const Bounds<double> b,
                       const arma::uword row_index,
                       const arma::uword col_index) : 
    Bounds<double>(b.lows, b.highs) {};

template <>
template <>
inline Bounds<double>::Bounds(const Bounds<arma::mat> b,
                       const arma::uword row_index,
                       const arma::uword col_index) : 
    Bounds<double>(b.lows(row_index, col_index),
                   b.highs(row_index, col_index)) {};

template <>
template <>
inline Bounds<arma::vec>::Bounds(const Bounds<double> b,
                          const arma::uword row_index,
                          const arma::uvec col_indicies) : 
    Bounds<arma::vec>(arma::ones<arma::vec>(col_indicies.n_elem)*b.lows,
                      arma::ones<arma::vec>(col_indicies.n_elem)*b.highs) {};

template <>
template <>
inline Bounds<arma::vec>::Bounds(const Bounds<arma::mat> b,
                          const arma::uword row_index,
                          const arma::uvec col_indicies) : 
    Bounds<arma::vec>(row_elem(b.lows, row_index, col_indicies),
                      row_elem(b.highs, row_index, col_indicies)) {};


template <class T, class Derived>
struct PenaltyBase
{
    const T l0;
    typedef T type;
    
    PenaltyBase(const T l0): l0{l0}{};
    
};


// template <class T>
// struct PenaltyL0 : public PenaltyBase<T, PenaltyL0<T>>
// {
//     PenaltyL0(const T l0): PenaltyBase<T, PenaltyL0<T>>(l0){};
//     
//     template<class B>
//     PenaltyL0<T> inline scale(const B two_a) const{
//         const T new_l0 = this->l0/two_a;
//         return PenaltyL0(new_l0);
//     }
//     
//     template<class O>
//     inline O cost(const O beta) const{
//         return MULT(this->l0, (beta != 0));
//     }
//     
// };


template <class T>
struct PenaltyL0L2 : public PenaltyBase<T, PenaltyL0L2<T>>
{
    const T l2;
    
    PenaltyL0L2(const T l0, const T l2) : PenaltyBase<T, PenaltyL0L2<T>>(l0), l2{l2} {};
    
    template <class TT>
    PenaltyL0L2(const PenaltyL0L2<TT>,
                const arma::uword row_index,
                const arma::uword col_index);
    
    template <class TT>
    PenaltyL0L2(const PenaltyL0L2<TT>,
                const arma::uword row_index,
                const arma::uvec col_indicies);

    template<class S>
    PenaltyL0L2<S> scale(const S two_a) const {
        return PenaltyL0L2<S>(this->l0/two_a, this->l2/two_a);
    }

    template<class O>
    inline O cost(const O beta) const{
        return MULT(this->l0,(beta != 0)) + MULT(this->l2,SQUARE(beta));
    }
    
};

template <>
template <>
inline PenaltyL0L2<double>::PenaltyL0L2(const PenaltyL0L2<double> p,
                                 const arma::uword row_index,
                                 const arma::uword col_index) : 
    PenaltyBase<double, PenaltyL0L2<double>>(p.l0), l2{p.l2} {};

template <>
template <>
inline PenaltyL0L2<double>::PenaltyL0L2(const PenaltyL0L2<arma::mat> p,
                                 const arma::uword row_index,
                                 const arma::uword col_index) : 
    PenaltyBase<double, PenaltyL0L2<double>>(
            p.l0(row_index, col_index)), 
            l2{p.l2(row_index, col_index)} {};

template <>
template <>
inline PenaltyL0L2<arma::vec>::PenaltyL0L2(const PenaltyL0L2<double> p,
                                    const arma::uword row_index,
                                    const arma::uvec col_indicies) : 
    PenaltyBase<arma::vec, PenaltyL0L2<arma::vec>>(
            arma::ones<arma::vec>(col_indicies.n_elem)*(p.l0)),
            l2{arma::ones<arma::vec>(col_indicies.n_elem)*(p.l2)} {};

template <>
template <>
inline PenaltyL0L2<arma::vec>::PenaltyL0L2(const PenaltyL0L2<arma::mat> p,
                                    const arma::uword row_index,
                                    const arma::uvec col_indicies) : 
    PenaltyBase<arma::vec, PenaltyL0L2<arma::vec>>(
            row_elem(p.l0, row_index, col_indicies)),
            l2{row_elem(p.l2, row_index, col_indicies)} {};


// TODO: Make PenaltyL0L1L2 inherit PenaltyL0, PenaltyL1, PenaltyL2
template <class T>
struct PenaltyL0L1L2 : public PenaltyBase<T, PenaltyL0L1L2<T>>
{
    const T l1;
    const T l2;
    
    PenaltyL0L1L2(const T l0, const T l1, const T l2) : PenaltyBase<T, PenaltyL0L1L2<T>>(l0), l1{l1}, l2{l2} {};
    
    template <class TT>
    PenaltyL0L1L2(const PenaltyL0L1L2<TT>,
                  const arma::uword row_index,
                  const arma::uword col_index);
    
    template <class TT>
    PenaltyL0L1L2(const PenaltyL0L1L2<TT>,
                  const arma::uword row_index,
                  const arma::uvec col_indicies);
    
    template<class S>
    PenaltyL0L1L2<S> scale(const S two_a) const{
        const S new_l0 = this->l0/two_a;
        const S new_l1 = this->l1/two_a;
        const S new_l2 = this->l2/two_a;
        return PenaltyL0L1L2<S>(new_l0, new_l1,new_l2);
    }
    
    template<class O>
    inline O cost(const O beta) const{
        return MULT(this->l0, (beta != 0)) + MULT(this->l1, ABS(beta)) + MULT(this->l2,SQUARE(beta));
    }
    
};

template <>
template <>
inline PenaltyL0L1L2<double>::PenaltyL0L1L2(const PenaltyL0L1L2<double> p,
                                     const arma::uword row_index,
                                     const arma::uword col_index) : 
    PenaltyBase<double, PenaltyL0L1L2<double>>(p.l0), l1{p.l1}, l2{p.l2} {};

template <>
template <>
inline PenaltyL0L1L2<double>::PenaltyL0L1L2(const PenaltyL0L1L2<arma::mat> p,
                                 const arma::uword row_index,
                                 const arma::uword col_index) : 
    PenaltyBase<double, PenaltyL0L1L2<double>>(
            p.l0(row_index, col_index)), 
            l1{p.l1(row_index, col_index)},
            l2{p.l2(row_index, col_index)} {};

template <>
template <>
inline PenaltyL0L1L2<arma::vec>::PenaltyL0L1L2(const PenaltyL0L1L2<double> p,
                                    const arma::uword row_index,
                                    const arma::uvec col_indicies) : 
    PenaltyBase<arma::vec, PenaltyL0L1L2<arma::vec>>(
            arma::ones<arma::vec>(col_indicies.n_elem)*(p.l0)),
            l1{arma::ones<arma::vec>(col_indicies.n_elem)*(p.l1)},
            l2{arma::ones<arma::vec>(col_indicies.n_elem)*(p.l2)} {};

template <>
template <>
inline PenaltyL0L1L2<arma::vec>::PenaltyL0L1L2(const PenaltyL0L1L2<arma::mat> p,
                                    const arma::uword row_index,
                                    const arma::uvec col_indicies) : 
    PenaltyBase<arma::vec, PenaltyL0L1L2<arma::vec>>(
            row_elem(p.l0, row_index, col_indicies)),
            l1{row_elem(p.l1, row_index, col_indicies)},
            l2{row_elem(p.l2, row_index, col_indicies)} {};


template <class P, class B>
struct Oracle{
    const P penalty;
    const B bounds;
    
    Oracle(const P penalty, const B bounds) : penalty{penalty}, bounds{bounds} {};
    
    template <class PP, class BB>
    Oracle(const PP penalty,
           const BB bounds,
           const arma::uword row_index,
           const arma::uword col_index) 
        : penalty{P(penalty, row_index, col_index)},
          bounds{B(bounds, row_index, col_index)} {};
    
    template <class PP, class BB>
    Oracle(const PP penalty,
           const BB bounds,
           const arma::uword row_index,
           const arma::uvec col_indicies) 
        : penalty{P(penalty, row_index, col_indicies)},
          bounds{B(bounds, row_index, col_indicies)} {};
    
    template <class T>
    inline T prox(const T theta) const{
        return _prox(theta, penalty, bounds);
    }
    
    template <class T>
    inline T Q(const T a, const T b) const {
        const T two_a = 2*a;
        const T b2 = -b/two_a;
        return _prox(b2, penalty.scale(two_a), bounds);
    }
    
    template <class T>
    inline std::tuple<T, T> Qobj(const T a, const T b) const{
        const T beta = Q(a, b);
        const T beta_cost = penalty.cost(beta);
        
        return {beta, beta_cost + MULT(b, beta) + MULT(a, SQUARE(beta))};
    }
    
    
};

template <template<class> class P, class E>
auto inline _get_scalar_oracle(const P<E> penalty,
                               const NoBounds bounds,
                               const size_t row_index,
                               const size_t col_index){
    return Oracle<P<double>, NoBounds>(penalty, bounds, row_index, col_index);
}

template <template<class> class P, class E>
auto inline _get_row_oracle(const P<E> penalty,
                            const NoBounds bounds,
                            const arma::uword row_index,
                            const arma::uvec col_indicies) {
    return Oracle<P<arma::vec>, NoBounds>(penalty, bounds, row_index, col_indicies);
}

template <template<class> class P, class E, class B>
auto inline _get_scalar_oracle(const P<E> penalty,
                               const Bounds<B> bounds,
                               const size_t row_index,
                               const size_t col_index){
    return Oracle<P<double>, Bounds<double>>(penalty, bounds, row_index, col_index);
}

template <template<class> class P, class E, class B>
auto inline _get_row_oracle(const P<E> penalty,
                            const Bounds<B> bounds,
                            const arma::uword row_index,
                            const arma::uvec col_indicies) {
    return Oracle<P<arma::vec>, Bounds<arma::vec>>(penalty, bounds, row_index, col_indicies);
}

template <class T, class P>
inline T _prox(const T theta, const PenaltyL0L2<P> penalty, const NoBounds b)
{
    const P two_l2_plus_1 = 2*penalty.l2 + 1;
    const P thresh = SQRT(2*MULT(penalty.l0,two_l2_plus_1));
    const T max_val = theta/two_l2_plus_1;
    const T abs_theta = ABS(theta);
    return MULT(max_val, abs_theta >= thresh);
}

template <class T, class P, class B>
inline T _prox(const T theta, const PenaltyL0L2<P> penalty, const Bounds<B> b)
{
    const P two_l2_plus_1 = 2*penalty.l2 + 1;
    const P two_l0 = 2*penalty.l0;
    const T theta_opt_no_l0 = theta/two_l2_plus_1;
    const T theta_opt_bounds = CLAMP(theta_opt_no_l0, b.lows, b.highs);
    const T delta = SQRT(MAX(SQUARE(theta) - MULT(two_l0,two_l2_plus_1), 0.))/two_l2_plus_1;
    return MULT(theta_opt_bounds,
                (ABS(theta_opt_no_l0) >= SQRT(two_l0/two_l2_plus_1) 
                     && (theta_opt_no_l0 - delta) <= theta_opt_bounds 
                     && (theta_opt_no_l0 + delta) >= theta_opt_bounds));
}

template <class T, class P>
inline T _prox(const T theta, const PenaltyL0L1L2<P> penalty, const NoBounds b)
{
    const P two_l2_plus_1 = 2*penalty.l2 + 1;
    const P two_l0 = 2*penalty.l0;
    const T abs_beta_m_l1 = ABS(theta) - penalty.l1;
    const T ns_beta_opt_no_l0 = abs_beta_m_l1/two_l2_plus_1;
    const T beta_opt_no_l0 = MULT(SIGN(theta),ns_beta_opt_no_l0);
    return MULT(beta_opt_no_l0,
                ns_beta_opt_no_l0 >= SQRT(two_l0/two_l2_plus_1));
}


template <class T, class P, class B>
inline T _prox(const T theta, const PenaltyL0L1L2<P> penalty, const Bounds<B> b)
{
    const P two_l2_plus_1 = 2*penalty.l2 + 1;
    const P two_l0 = 2*penalty.l0;
    const T abs_theta_m_l1 = ABS(theta) - penalty.l1;
    const T ns_theta_opt_no_l0 = abs_theta_m_l1/two_l2_plus_1;
    const T theta_opt_no_l0 = MULT(SIGN(theta),ns_theta_opt_no_l0);
    const T theta_opt_bounds = CLAMP(theta_opt_no_l0, b.lows, b.highs);
    const T delta = SQRT(MAX(SQUARE(abs_theta_m_l1) - MULT(two_l0,two_l2_plus_1), 0.))/two_l2_plus_1;
    return MULT(theta_opt_bounds,
                (ns_theta_opt_no_l0 >= SQRT(two_l0/two_l2_plus_1) 
                     && (theta_opt_no_l0 - delta) <= theta_opt_bounds 
                     && (theta_opt_no_l0 + delta) >= theta_opt_bounds));

}

#endif // ORACLE2_H