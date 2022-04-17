#include "utils.h"

std::vector<std::size_t> nth_largest_indices(const std::vector<double> &x,
                                             const std::size_t n) {
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
