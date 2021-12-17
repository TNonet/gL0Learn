#ifndef UTILS_H
#define UTILS_H
#include <iterator>
#include <functional>
#include "RcppArmadillo.h"

typedef std::tuple<arma::uword, arma::uword> coordinate;
typedef std::vector<coordinate> coordinate_vector;

inline coordinate inc(const coordinate c, const arma::uword p){
    const double i = std::get<0>(c);
    const double j = std::get<1>(c);
    if (std::get<1>(c) < p - 1){
        return {i, j+1};
    } else if (std::get<0>(c) < p - 1){
        return {i+1, 0};
    } else {
        Rcpp::Rcout << "Cannot increment coordinate (" << i << ", " << j << ")as it is already at maximium";
        Rcpp::stop("Error in coordinate inc");
    }
    
};


inline arma::vec row_elem(const arma::mat a, const arma::uword row, const arma::uvec indices){
    const arma::vec a_row = a.row(row);
    return a_row(indices);
}

inline arma::vec ABS(const arma::vec x){
    return arma::abs(x);
}

inline double ABS(const double x){
    return std::abs(x);
}


inline arma::vec SQRT(const arma::vec x){
    return arma::sqrt(x);
}


inline double SQRT(const double x){
    return std::sqrt(x);
}

template<class T>
inline T SQUARE(const T x){
    return arma::square(x);
}

template <>
inline double SQUARE(const double x){
    return x*x;
}


inline double MULT(const double x1, const double x2){
    return x1*x2;
}

inline arma::vec MULT(const arma::vec x1, const double x2){
    return x1*x2;
}

inline arma::vec MULT(const double x1, const arma::vec x2){
    return x1*x2;
}

inline arma::vec MULT(const arma::uvec x1, const double x2){
    return arma::conv_to<arma::vec>::from(x1)*x2;
}

inline arma::vec MULT(const double x1, const arma::uvec x2){
    return MULT(x2, x1);
}

inline arma::vec MULT(const arma::vec x1, const arma::vec x2){
    return x1%x2;
}

inline arma::vec MULT(const arma::uvec x1, const arma::vec x2){
    return x1%x2;
}

inline arma::vec MULT(const arma::vec x1, const arma::uvec x2){
    return x1%x2;
}

template<class T>
inline T SIGN(const T x){
    return arma::sign(x);
}

inline int SIGN(const double x){
    return (0. < x) - (x < 0.);
}


inline double CLAMP(const double x, const double lows, const double highs) {
    // -O3 Compiler should remove branches
    if (x < lows) 
        return lows;
    if (x > highs) 
        return highs;
    return x;
}

inline arma::vec CLAMP(const arma::vec x, const double lows, const double highs){
    return arma::clamp(x, lows, highs);
}

inline arma::vec CLAMP(const arma::vec x, const arma::vec lows, const arma::vec highs){
    const std::size_t n = x.n_rows;
    arma::vec x_clamped(n);
    for (std::size_t i = 0; i < n; i++){
        x_clamped.at(i) = CLAMP(x.at(i), lows.at(i), highs.at(i));
    }
    return x_clamped;
}

inline arma::vec MAX(const arma::vec x1, const double x2){
    return arma::max(x1, x2*arma::ones<arma::vec>(arma::size(x1)));
}


template<class T>
inline T MAX(const T x1, const T x2){
    return arma::max(x1, x2);
}


inline double MAX(const double x1, const double x2){
    return std::max(x1, x2);
}



#endif // UTILS_H