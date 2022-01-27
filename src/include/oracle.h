#ifndef ORACLE_H
#define ORACLE_H
#include "RcppArmadillo.h"
#include "utils.h"


template <class T>
T inline R_nl(const T & a, const T & b){
    return (1 + arma::sqrt(1 + 4 % a % b)) / (2*a);
}


double inline R_nl(const double a, const double b){
    return (1 + std::sqrt(1 + 4 * a * b)) / (2 * a);
}


template <size_t N, typename... Args>
decltype(auto) inline variadic_get(Args&&... as) noexcept {
    return std::get<N>(std::forward_as_tuple(as...));
}

template <typename... Args,
          typename = typename std::enable_if<sizeof...(Args) == 0>::type>
arma::mat inline get_by_var_args(const arma::mat& x, const Args... getitems){
    return x;
}

template <typename... Args,
          typename = typename std::enable_if<sizeof...(Args) == 0>::type>
arma::vec inline get_by_var_args(const arma::vec& x, const Args... getitems){
    return x;
}

template <typename... Args>
double inline get_by_var_args(const double x, const Args... getitems){
    return x;
}

template <typename... Args,
          typename = typename std::enable_if<sizeof...(Args) == 1>::type>
arma::vec inline get_by_var_args(const arma::mat& x, const Args... getitems){
    const arma::uword i = variadic_get<0>(getitems...);
    return x.col(i);
}


template <typename... Args,
          typename = typename std::enable_if<sizeof...(Args) == 2>::type>
double inline get_by_var_args(const arma::mat& x, const Args... getitems){
    const arma::uword i = variadic_get<0>(getitems...);
    const arma::uword j = variadic_get<1>(getitems...);
    return x(i, j);
}

template <typename T>
struct PenaltyL0
{
    const T l0;
    
    PenaltyL0(const T& l0): l0{l0}{};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL0<decltype(get_by_var_args(this->l0, getitems...))>(
                get_by_var_args(this->l0, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return MULT(get_by_var_args(this->l0, getitems...),
                    get_by_var_args(beta, getitems...) != 0);
    }
    
};

template <typename T>
struct PenaltyL1
{
    const T l1;
    
    PenaltyL1(const T& l1): l1{l1}{};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL1<decltype(get_by_var_args(this->l1, getitems...))>(
                get_by_var_args(this->l1, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return MULT(get_by_var_args(this->l1, getitems...),
                    ABS(get_by_var_args(beta, getitems...)));
    }
    
};

template <typename T>
struct PenaltyL2
{
    const T l2;
    
    PenaltyL2(const T& l2): l2{l2}{};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL2<decltype(get_by_var_args(this->l2, getitems...))>(
                get_by_var_args(this->l2, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return MULT(get_by_var_args(this->l2, getitems...),
                    SQUARE(get_by_var_args(beta, getitems...)));
    }
    
};

template <typename T>
struct PenaltyL0L2 : public PenaltyL0<T>, public PenaltyL2<T>
{
    PenaltyL0L2(const T& l0, const T& l2): PenaltyL0<T>(l0), PenaltyL2<T>(l2){};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL0L2<decltype(get_by_var_args(this->l0, getitems...))>(
                get_by_var_args(this->l0, getitems...),
                get_by_var_args(this->l2, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return PenaltyL0<T>::cost(beta, getitems...) + PenaltyL2<T>::cost(beta, getitems...);
    }
    
};

template <typename T>
struct PenaltyL0L1 : public PenaltyL0<T>, public PenaltyL1<T>
{
    PenaltyL0L1(const T& l0, const T& l1): PenaltyL0<T>(l0), PenaltyL1<T>(l1){};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL0L1<decltype(get_by_var_args(this->l0, getitems...))>(
                get_by_var_args(this->l0, getitems...),
                get_by_var_args(this->l1, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return PenaltyL0<T>::cost(beta, getitems...) + PenaltyL1<T>::cost(beta, getitems...);
    }
};

template <typename T>
struct PenaltyL0L1L2 : public PenaltyL0<T>, public PenaltyL1<T>, public PenaltyL2<T>
{
    PenaltyL0L1L2(const T& l0, const T& l1, const T& l2): PenaltyL0<T>(l0), PenaltyL1<T>(l1), PenaltyL2<T>(l2){};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return PenaltyL0L1L2<decltype(get_by_var_args(this->l0, getitems...))>(
                get_by_var_args(this->l0, getitems...),
                get_by_var_args(this->l1, getitems...),
                get_by_var_args(this->l2, getitems...));
    }
    
    template <typename U, typename... Args>
    inline auto cost(const U& beta,
                     const Args... getitems) const{
        return PenaltyL0<T>::cost(beta, getitems...) + PenaltyL1<T>::cost(beta, getitems...) + PenaltyL2<T>::cost(beta, getitems...);
    }
};



struct NoBounds {
    NoBounds() {};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return NoBounds();
    }
};

template <typename T>
struct Bounds{
    const T lows;
    const T highs;
    
    Bounds(const Bounds &b) : lows{b.lows}, highs{b.highs} {};
    
    Bounds(const T& lows, const T& highs) : lows{lows}, highs{highs} {};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return Bounds<decltype(get_by_var_args(this->lows, getitems...))>(
                get_by_var_args(this->lows, getitems...),
                get_by_var_args(this->highs, getitems...));
    }
    
};



template <class P, class B>
struct Oracle{
    const P penalty;
    const B bounds;
    
    Oracle(const P& penalty, const B& bounds) : penalty{penalty}, bounds{bounds} {};
    
    template <typename... Args>
    inline auto operator()(const Args... getitems) const {
        return Oracle<decltype(this->penalty(getitems...)), decltype(this->bounds(getitems...))>(
                this->penalty(getitems...),
                this->bounds(getitems...));
    }
    
    template <class T, typename... Args>
    inline auto prox(const T& theta,
                     const Args... getitems) const{
        return _prox(theta, this->penalty, this->bounds, getitems...);
    }
    
    
    template <class T, typename... Args>
    inline auto Q(const T& a,
                  const T& b,
                  const Args... getitems) const {
        const T two_a = 2*a;
        const T b2 = -b/two_a;
        return _prox(b2, this->penalty, this->bounds, two_a, getitems...);
    }
    
    template <class T, typename... Args>
    inline auto Qobj(const T& a,
                     const T& b,
                     const Args... getitems) const {
        const auto beta = this->Q(a, b, getitems...);
        const auto beta_cost = this->penalty.cost(beta, getitems...);
        return std::make_tuple(beta, (beta_cost 
                                          + MULT(get_by_var_args(b, getitems...), beta) 
                                          + MULT(get_by_var_args(a, getitems...), SQUARE(beta))));
    }
};

/*
 * TODO: _prox_wrapper is unecessary.
 * 
 *  Instead create prox to accept Penalty and NoBounds and getitems. Then 
 *  
 *  For each P in [L0, L1, L2, L0L1, L0L2, L1L1, L0L1L2]:
 *      For each B in [Bounds, NoBounds]:
 *          define proxP(theta, P, B, getitems...)
 *          define proxP(theta, P, B, scale, getitems...)
 *          define _proxP(theta, ...)
 * 
 */

template <class T, class P>
inline T _proxL0(const T& theta,
                 const P& l0)
{   
    return MULT(theta, ABS(theta) >= SQRT(2*l0));
}

template <class T, class P, class B>
inline T _proxL0(const T &theta, 
                 const P& l0, 
                 const B& lows,
                 const B& highs)
{
    const P two_l0 = 2*l0;
    const T theta_opt_bounds = CLAMP(theta, lows, highs);
    const T delta = SQRT(MAX(SQUARE(theta) - two_l0, 0.));
    return MULT(theta_opt_bounds,
                (ABS(theta) >= SQRT(two_l0) 
                     && (theta - delta) <= theta_opt_bounds 
                     && (theta + delta) >= theta_opt_bounds));
}

template <class T, class P, typename... Args>
inline auto _prox(const T& theta,
                  const PenaltyL0<P>& penalty,
                  const NoBounds bounds,
                  const Args... getitems){
    return _proxL0(get_by_var_args(theta, getitems...),
                   get_by_var_args(penalty.l0, getitems...));
    
}

template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta,
                  const PenaltyL0<P>& penalty,
                  const Bounds<B> bounds,
                  const Args... getitems){
    return _proxL0(get_by_var_args(theta, getitems...),
                   get_by_var_args(penalty.l0, getitems...),
                   get_by_var_args(bounds.lows, getitems...),
                   get_by_var_args(bounds.highs, getitems...));
    
}

template <class T, class P, typename... Args>
inline auto _prox(const T& theta,
                  const PenaltyL0<P>& penalty,
                  const NoBounds bounds,
                  const T& scale,
                  const Args... getitems){
    return _proxL0(get_by_var_args(theta, getitems...),
                   eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)));
    
}

template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta,
                  const PenaltyL0<P>& penalty,
                  const Bounds<B> bounds,
                  const T& scale,
                  const Args... getitems){
    return _proxL0(get_by_var_args(theta, getitems...),
                   eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)),
                   get_by_var_args(bounds.lows, getitems...),
                   get_by_var_args(bounds.highs, getitems...));
    
}


// L0L2 4 wrappers
template <class T, class P>
inline T _proxL0L2(const T& theta, const P& l0, const P& l2)
{
    const P two_l2_plus_1 = 2*l2 + 1;
    return MULT(theta/two_l2_plus_1,  ABS(theta) >= SQRT(2*MULT(l0,two_l2_plus_1)));
}


template <class T, class P, class B>
inline T _proxL0L2(const T &theta, const P& l0, const P& l2, const B& lows, const B& highs)
{
    const P two_l2_plus_1 = 2*l2 + 1;
    const P two_l0 = 2*l0;
    const T theta_opt_no_l0 = theta/two_l2_plus_1;
    const T theta_opt_bounds = CLAMP(theta_opt_no_l0, lows, highs);
    const T delta = SQRT(MAX(SQUARE(theta) - MULT(two_l0,two_l2_plus_1), 0.))/two_l2_plus_1;
    return MULT(theta_opt_bounds,
                (ABS(theta_opt_no_l0) >= SQRT(two_l0/two_l2_plus_1) 
                     && (theta_opt_no_l0 - delta) <= theta_opt_bounds 
                     && (theta_opt_no_l0 + delta) >= theta_opt_bounds));
}


template <class T, class P, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L2<P>& penalty,
                  const NoBounds& b,
                  const Args... getitems){
    return _proxL0L2(get_by_var_args(theta, getitems...),
                     get_by_var_args(penalty.l0, getitems...),
                     get_by_var_args(penalty.l2, getitems...));
}

template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L2<P>& penalty,
                  const Bounds<B>& b,
                  const Args... getitems){
    return _proxL0L2(get_by_var_args(theta, getitems...),
                     get_by_var_args(penalty.l0, getitems...),
                     get_by_var_args(penalty.l2, getitems...),
                     get_by_var_args(b.lows, getitems...),
                     get_by_var_args(b.highs, getitems...));
}

template <class T, class P, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L2<P>& penalty,
                  const NoBounds& b,
                  const T& scale,
                  const Args... getitems){
    return _proxL0L2(get_by_var_args(theta, getitems...),
                     eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)),
                     eval(get_by_var_args(penalty.l2, getitems...)/get_by_var_args(scale, getitems...)));
}

template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L2<P>& penalty,
                  const Bounds<B>& b,
                  const T& scale,
                  const Args... getitems){
    return _proxL0L2(get_by_var_args(theta, getitems...),
                     eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)),
                     eval(get_by_var_args(penalty.l2, getitems...)/get_by_var_args(scale, getitems...)),
                     get_by_var_args(b.lows, getitems...),
                     get_by_var_args(b.highs, getitems...));
}


template <class T, class P>
inline T _proxL0L1L2(const T& theta, const P& l0, const P& l1, const P& l2)
{
    const P two_l2_plus_1 = 2*l2 + 1;
    const T ns_beta_opt_no_l0 = (ABS(theta) - l1)/two_l2_plus_1;
    const T beta_opt_no_l0 = MULT(SIGN(theta),ns_beta_opt_no_l0);
    return MULT(beta_opt_no_l0,
                ns_beta_opt_no_l0 >= SQRT(2*l0/two_l2_plus_1));
}


template <class T, class P, class B>
inline T _proxL0L1L2(const T& theta, const P& l0, const P& l1, const P& l2, const B& lows, const B& highs)
{
    const P two_l2_plus_1 = 2*l2 + 1;
    const P two_l0 = 2*l0;
    const T abs_theta_m_l1 = ABS(theta) - l1;
    const T ns_theta_opt_no_l0 = abs_theta_m_l1/two_l2_plus_1;
    const T theta_opt_no_l0 = MULT(SIGN(theta),ns_theta_opt_no_l0);
    const T theta_opt_bounds = CLAMP(theta_opt_no_l0, lows, highs);
    const T delta = SQRT(MAX(SQUARE(abs_theta_m_l1) - MULT(two_l0,two_l2_plus_1), 0.))/two_l2_plus_1;
    return MULT(theta_opt_bounds,
                (ns_theta_opt_no_l0 >= SQRT(two_l0/two_l2_plus_1) 
                     && (theta_opt_no_l0 - delta) <= theta_opt_bounds 
                     && (theta_opt_no_l0 + delta) >= theta_opt_bounds));
}


template <class T, class P, typename... Args>
inline auto _prox(const T& theta,
                  const PenaltyL0L1L2<P>& penalty,
                  const NoBounds& b,
                  const Args... getitems){
    return _proxL0L1L2(get_by_var_args(theta, getitems...),
                       get_by_var_args(penalty.l0, getitems...),
                       get_by_var_args(penalty.l1, getitems...),
                       get_by_var_args(penalty.l2, getitems...));
}


template <class T, class P, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L1L2<P>& penalty,
                  const NoBounds& b,
                  const T& scale,
                  const Args... getitems){
    return _proxL0L1L2(get_by_var_args(theta, getitems...),
                       eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)),
                       eval(get_by_var_args(penalty.l1, getitems...)/get_by_var_args(scale, getitems...)),
                       eval(get_by_var_args(penalty.l2, getitems...)/get_by_var_args(scale, getitems...)));
}


template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L1L2<P>& penalty,
                  const Bounds<B>& b,
                  const Args... getitems){
    return _proxL0L1L2(get_by_var_args(theta, getitems...),
                       get_by_var_args(penalty.l0, getitems...),
                       get_by_var_args(penalty.l1, getitems...),
                       get_by_var_args(penalty.l2, getitems...),
                       get_by_var_args(b.lows, getitems...),
                       get_by_var_args(b.highs, getitems...));
}


template <class T, class P, class B, typename... Args>
inline auto _prox(const T& theta, 
                  const PenaltyL0L1L2<P>& penalty,
                  const Bounds<B>& b,
                  const T& scale,
                  const Args... getitems){
    return _proxL0L1L2(get_by_var_args(theta, getitems...),
                       eval(get_by_var_args(penalty.l0, getitems...)/get_by_var_args(scale, getitems...)),
                       eval(get_by_var_args(penalty.l1, getitems...)/get_by_var_args(scale, getitems...)),
                       eval(get_by_var_args(penalty.l2, getitems...)/get_by_var_args(scale, getitems...)),
                       get_by_var_args(b.lows, getitems...),
                       get_by_var_args(b.highs, getitems...));
}


#endif // ORACLE_H