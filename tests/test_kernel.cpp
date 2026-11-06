// skmd-lammps/tests/test_kernel.cpp
// Copyright (C) 2025-2026 Fraser Birks. GPL-2.0, see LICENSE.
// Standalone tests for skmd_kernel.h — no LAMMPS linkage required.
// Build: cmake -DSKMD_TESTING=ON, then cmake --build . --target test_kernel
// Run:   ./test_kernel

#include "skmd_kernel.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++tests_run;                                                           \
    if (cond) {                                                            \
      ++tests_passed;                                                    \
    } else {                                                               \
      std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

int main()
{
  // 1. Kernel at zero distance equals a
  {
    double q[4] = {1.0, -2.0, 0.5, 3.0};
    double a = 2.5, b = 1.0;
    CHECK_NEAR(skmd_kernel(q, q, 4, a, b), a, 1e-12);
  }

  // 2. Kernel decreases monotonically with distance (1-D)
  {
    double q0[1] = {0.0};
    double q1[1] = {1.0};
    double q2[1] = {2.0};
    double q3[1] = {3.0};
    double a = 1.0, b = 1.0;
    double k0 = skmd_kernel(q0, q0, 1, a, b);
    double k1 = skmd_kernel(q0, q1, 1, a, b);
    double k2 = skmd_kernel(q0, q2, 1, a, b);
    double k3 = skmd_kernel(q0, q3, 1, a, b);
    CHECK(k0 > k1);
    CHECK(k1 > k2);
    CHECK(k2 > k3);
  }

  // 3. Kernel derivative matches finite difference (3-D)
  {
    double q[3]  = { 0.5, -0.3,  1.2};
    double qp[3] = {-0.1,  0.7,  0.4};
    double a = 2.0, b = 0.8;
    double deriv[3];
    skmd_kernel_deriv(q, qp, 3, a, b, deriv);

    const double eps = 1e-6;
    double qph[3] = {q[0], q[1], q[2]};
    for (int i = 0; i < 3; ++i) {
      qph[i] = q[i] + eps;
      double fd = (skmd_kernel(qph, qp, 3, a, b) - skmd_kernel(q, qp, 3, a, b)) / eps;
      qph[i] = q[i];
      CHECK_NEAR(deriv[i], fd, 1e-5);
    }
  }

  // 4. median_pairwise_distance: points 0,1,3 -> distances 1,2,3 -> median 2
  {
    double row0[1] = {0.0};
    double row1[1] = {1.0};
    double row2[1] = {3.0};
    const double* ptrs[3] = {row0, row1, row2};
    CHECK_NEAR(median_pairwise_distance(ptrs, 3, 1), 2.0, 1e-12);
  }

  // 5. median_pairwise_distance: single point returns 0
  {
    double row0[2] = {1.0, 2.0};
    const double* ptrs[1] = {row0};
    CHECK_NEAR(median_pairwise_distance(ptrs, 1, 2), 0.0, 1e-12);
  }

  // 6. median_pairwise_distance, even pair count: points 0,1,2,4 -> 1,1,2,2,3,4 -> 2
  {
    double r0[1]={0.0}, r1[1]={1.0}, r2[1]={2.0}, r3[1]={4.0};
    const double* ptrs[4] = {r0, r1, r2, r3};
    CHECK_NEAR(median_pairwise_distance(ptrs, 4, 1), 2.0, 1e-12);
  }

  std::printf("%d/%d tests passed\n", tests_passed, tests_run);
  return (tests_passed == tests_run) ? 0 : 1;
}
