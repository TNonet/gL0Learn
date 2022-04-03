#ifndef UTILS_H
#define UTILS_H
#include "arma_includes.h"
#include <functional>
#include <iterator>
#include <queue>

inline arma::vec row_elem(const arma::mat &a, const arma::uword row,
                          const arma::uvec &indices) {
  const arma::vec a_row = a.row(row);
  return a_row(indices);
}

inline double eval(const double x) { return x; }

template <typename T> inline auto eval(const T &x) { return x.eval(); }

std::vector<std::size_t> inline nth_largest_indices(
    const std::vector<double> &x, const std::size_t n) {
  std::priority_queue<std::pair<double, size_t>> q;

  auto it = x.begin();
  const auto it_end = x.end();
  for (size_t i = 0; it != it_end; ++i, ++it) {
    q.push(std::pair<double, int>(*it, i));
  }

  std::vector<std::size_t> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    indices.push_back(q.top().second);
    q.pop();
  }

  return indices;
}

// inline arma::vec ABS(const arma::vec& x){
//     return arma::abs(x);
// }

inline arma::mat ADD(const arma::mat &x1, const arma::mat &x2) {
  return x1 + x2;
}

inline double ADD(const double x1, const double x2) { return x1 + x2; }

inline arma::vec ADD(const arma::vec &x1, const arma::vec &x2) {
  return x1 + x2;
}

inline arma::mat ADD(const arma::mat &x1, const arma::mat &x2,
                     const arma::mat &x3) {
  return x1 + x2 + x3;
}

inline double ADD(const double x1, const double x2, const double x3) {
  return x1 + x2 + x3;
}

inline arma::vec ADD(const arma::vec &x1, const arma::vec &x2, arma::vec &x3) {
  return x1 + x2 + x3;
}

inline arma::mat ABS(const arma::mat &x) { return arma::abs(x); }

inline double ABS(const double x) { return std::abs(x); }

// inline arma::vec SQRT(const arma::vec& x){
//     return arma::sqrt(x);
// }

inline arma::mat SQRT(const arma::mat &x) { return arma::sqrt(x); }

inline double SQRT(const double x) { return std::sqrt(x); }

inline arma::mat SQUARE(const arma::mat &x) { return arma::square(x); }

inline double SQUARE(const double x) { return x * x; }

inline double MULT(const double x1, const double x2) { return x1 * x2; }

inline arma::mat MULT(const arma::mat &x1, const double x2) { return x1 * x2; }

inline arma::mat MULT(const double x1, const arma::mat &x2) { return x1 * x2; }

inline arma::mat MULT(const arma::umat &x1, const double x2) {
  return arma::conv_to<arma::mat>::from(x1) * x2;
}

inline arma::mat MULT(const double x1, const arma::umat &x2) {
  return MULT(x2, x1);
}

inline arma::mat MULT(const arma::mat &x1, const arma::mat &x2) {
  return x1 % x2;
}

inline arma::mat MULT(const arma::umat &x1, const arma::mat &x2) {
  return x1 % x2;
}

inline arma::mat MULT(const arma::mat &x1, const arma::umat &x2) {
  return x1 % x2;
}

inline arma::mat DIVIDE(const arma::mat &x1, const arma::mat &x2) {
  return x1 / x2;
}

inline arma::mat DIVIDE(const arma::mat &x1, const double &x2) {
  return x1 / x2;
}

inline double DIVIDE(const double x1, const double x2) { return x1 / x2; }

inline arma::mat DIVIDE(const double x1, const arma::mat &x2) {
  return x1 / x2;
}

inline arma::mat SIGN(const arma::mat &x) { return arma::sign(x); }

inline int SIGN(const double x) { return (0. < x) - (x < 0.); }

inline arma::umat not_eq_zero(const arma::mat &x) { return x != 0; }

inline arma::uvec not_eq_zero(const arma::vec &x) { return x != 0; }

inline std::size_t not_eq_zero(const double x) { return x != 0; }

inline double CLAMP(const double x, const double lows, const double highs) {
  // -O3 Compiler should remove branches
  if (x < lows)
    return lows;
  if (x > highs)
    return highs;
  return x;
}

inline arma::mat CLAMP(const arma::mat &x, const double lows,
                       const double highs) {
  return arma::clamp(x, lows, highs);
}

inline arma::mat CLAMP(const arma::mat &x, const arma::mat &lows,
                       const arma::mat &highs) {

  arma::mat x_clamped(arma::size(x));

  arma::mat::iterator x_clamped_it = x_clamped.begin();
  arma::mat::iterator x_clamped_end = x_clamped.end();

  arma::mat::const_iterator x_it = x.begin();
  arma::mat::const_iterator lows_it = lows.begin();
  arma::mat::const_iterator highs_it = highs.begin();

  for (; x_clamped_it != x_clamped_end;
       ++x_it, ++lows_it, ++highs_it, ++x_clamped_it) {
    (*x_clamped_it) = CLAMP((*x_it), (*lows_it), (*highs_it));
  }

  return x_clamped;
}

inline arma::mat MAX(const arma::mat &x1, const double x2) {
  return arma::max(x1, x2 * arma::ones<arma::mat>(arma::size(x1)));
}

template <class T> inline T MAX(const T &x1, const T &x2) {
  return arma::max(x1, x2);
}

inline double MAX(const double x1, const double x2) { return std::max(x1, x2); }

#endif // UTILS_H