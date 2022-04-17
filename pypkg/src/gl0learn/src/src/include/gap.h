#ifndef H_GAP
#define H_GAP
#include "arma_includes.h"

enum GapMethod : uint8_t { both = 0, first = 1, second = 2 };

double relative_gap(double old_objective, double new_objective,
                    GapMethod method, bool one_normalize);

#endif // H_GAP