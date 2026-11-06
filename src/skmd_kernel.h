/* ----------------------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

inline double skmd_kernel(
    const double* q,
    const double* q_prime,
    int n,
    double a,
    double b)
{
  double diff2 = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = q[i] - q_prime[i];
    diff2 += d * d;
  }
  return a * std::exp(-diff2 / (2.0 * b * b));
}

inline void skmd_kernel_deriv(
    const double* q,
    const double* q_prime,
    int n,
    double a,
    double b,
    double* deriv_out)
{
  double fac = -1.0 / (b * b);
  double kval = skmd_kernel(q, q_prime, n, a, b);
  for (int i = 0; i < n; ++i)
    deriv_out[i] = fac * kval * (q[i] - q_prime[i]);
}

// Median of all pairwise Euclidean distances between J descriptor vectors of
// length ncoeff. Returns 0 if J < 2.
inline double median_pairwise_distance(const double* const* q, int J, int ncoeff)
{
  if (J < 2) return 0.0;

  const std::size_t npairs = static_cast<std::size_t>(J) * (J - 1) / 2;
  std::vector<double> dists;
  dists.reserve(npairs);

  for (int i = 0; i < J; ++i) {
    for (int j = i + 1; j < J; ++j) {
      double s = 0.0;
      for (int c = 0; c < ncoeff; ++c) {
        const double dx = q[i][c] - q[j][c];
        s += dx * dx;
      }
      dists.push_back(std::sqrt(s));
    }
  }

  const std::size_t mid = npairs / 2;
  if (npairs % 2 == 1) {
    std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
    return dists[mid];
  } else {
    std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
    const double hi = dists[mid];
    std::nth_element(dists.begin(), dists.begin() + (mid - 1), dists.begin() + mid);
    const double lo = dists[mid - 1];
    return 0.5 * (lo + hi);
  }
}
