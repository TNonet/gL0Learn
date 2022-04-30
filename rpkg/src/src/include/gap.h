#ifndef H_GAP
#define H_GAP
#include "arma_includes.h"

enum GapMethod : uint8_t { both = 0, first = 1, second = 2 };

const double INF = std::numeric_limits<double>::infinity();

double inline relative_gap(const double old_objective,
                           const double new_objective, const GapMethod method,
                           const bool one_normalize) {
  // Does the sign of this value matter?
  // ONly usage of this function is prompt made positive with an ABS
  if ((old_objective == INF) || (new_objective == -INF)) {
    return 1.;
  } else if ((old_objective == -INF) || (new_objective == +INF)) {
    return -1.;
  }

  double benchmark;
  if (method == GapMethod::both) {
    benchmark = std::max(std::abs(old_objective), std::abs(new_objective));
  } else if (method == GapMethod::first) {
    benchmark = std::abs(old_objective);
  } else { // method == GapMethod::second
    benchmark = std::abs(new_objective);
  }

  if (one_normalize) {
    benchmark = std::max(benchmark, 1.0);
  }

  return (old_objective - new_objective) / benchmark;
}

#endif // H_GAP