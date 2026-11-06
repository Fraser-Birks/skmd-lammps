/* ----------------------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(skmd,FixSKMD);
// clang-format on
#else

#ifndef LMP_FIX_SKMD_H
#define LMP_FIX_SKMD_H

#include "fix.h"

namespace LAMMPS_NS {

class Compute;

class FixSKMD : public Fix {
 public:
  FixSKMD(class LAMMPS *, int, char **);
  ~FixSKMD() override;

  int setmask() override;
  void init() override;

  // Fix hooks
  void post_force(int vflag) override;
  void end_of_step() override;
  void post_run() override;
  double compute_scalar() override;
  double compute_vector(int) override;


 protected:
  // helpers
  void allocate();
  void read_files(char *dataset_base);

  double k(const double *q, const double *q_prime) const;
  void compute_k_deriv(const double *q, const double *q_prime);

  void add_current_point_to_dataset();

 protected:
  // computes
  Compute *compute_q_ptr = nullptr;
  Compute *compute_jac_ptr = nullptr;
  char *compute_q_id = nullptr;
  char *compute_jac_id = nullptr;

  // coeff data
  int ncoeff = 0;

  // kernel params
  double a;
  double b;
  double beta = 1.0;  // weight of the attractive term (keyword `beta`)

  // threshold stopping
  double thresh;
  double fnorm2;
  int buffer_steps = 0;
  bigint run_start_step = 0;

  // running mean of fnorm2 over the run; the last-step value is very noisy
  // once b is below typical descriptor distances
  double fnorm2_sum = 0.0;
  bigint fnorm2_count = 0;

  // dataset bookkeeping
  int dataset_size = 0;
  int alloc_rows = 1;
  int dataset_N = 0;  // number of atoms implied by dataset (from jacs rows)
  int dataset_N_alloc = 0;
  int ncoeff_alloc = 0;
  bool dataset_allocated = false;

  // dataset arrays
  double **qjs = nullptr;    // [alloc_rows][ncoeffall]
  double ***fjs = nullptr;   // [alloc_rows][dataset_N][3]
  double ***jacs = nullptr;  // [alloc_rows][3*dataset_N][ncoeffall]

  // runtime buffers
  double *q1 = nullptr;       // [ncoeffall]
  double *deriv_vec = nullptr;// [ncoeffall]
  double *Fskmdx = nullptr;   // [3*dataset_N]

  // state
  bool allocated = false;
  bool q_computed = false;
  bool forces_stored = false;
  bool thresh_flag = false;
  bool triggered = false;

  // dataset path base
  char *dataset_filename = nullptr;
};

} // namespace LAMMPS_NS

#endif
#endif
