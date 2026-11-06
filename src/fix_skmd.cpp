/* ----------------------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

#include "fix_skmd.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "output.h"
#include "thermo.h"
#include "tokenizer.h"
#include "update.h"
#include "timer.h"

#include "skmd_kernel.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;


// file-scope helper (must be visible before FixSKMD::read_files uses it)
static inline void ensure_file_exists(const std::string &fname,
                                      const std::string &header)
{
  std::ofstream ofs(fname, std::ios::app);
  if (!ofs.good())
    throw std::runtime_error("Failed to create dataset file: " + fname);

  if (ofs.tellp() == 0)
    ofs << header << "\n";
}


// -----------------------------------------------------------------------------

FixSKMD::FixSKMD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  compute_q_id = nullptr;
  compute_jac_id = nullptr;
  dataset_filename = nullptr;

  // runtime arrays
  q1 = deriv_vec = nullptr;
  Fskmdx = nullptr;

  qjs = nullptr;
  fjs = nullptr;
  jacs = nullptr;

  allocated = false;
  q_computed = false;

  dataset_size = 0;
  alloc_rows = 1;
  dataset_N = 0;           // atoms implied by dataset jacobians (0 until known)
  dataset_allocated = false;

  thresh = 0.0;
  thresh_flag = false;

  double b_max_val = 0.0, b_min_val = 0.0;
  bool b_max_flag = false, b_min_flag = false;

  // fix id group_id skmd c_q_id c_jac_id dataset_fname a b_or_adaptive [keywords]
  if (narg < 8)
    error->all(FLERR, "Illegal fix skmd command: not enough arguments");

  compute_q_id = utils::strdup(arg[3]);
  compute_jac_id = utils::strdup(arg[4]);
  dataset_filename = utils::strdup(arg[5]);

  a = utils::numeric(FLERR, arg[6], false, lmp);

  bool b_adaptive = false;
  double b_input = 0.0;

  if (strcmp(arg[7], "adaptive") == 0) {
    b_adaptive = true;
  } else {
    b_input = utils::numeric(FLERR, arg[7], false, lmp);
    if (!(b_input > 0.0))
      error->all(FLERR, "Illegal fix skmd command: b must be > 0 or 'adaptive'");
  }

  int iarg = 8;
  // optional keyword-value pairs
  while (iarg < narg) {
    if (strcmp(arg[iarg], "thresh") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for 'thresh' in fix skmd");

      thresh = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      thresh_flag = true;

      iarg += 2;
    } else if (strcmp(arg[iarg], "b_max") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for 'b_max' in fix skmd");

      b_max_val = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      b_max_flag = true;

      iarg += 2;
    } else if (strcmp(arg[iarg], "b_min") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for 'b_min' in fix skmd");

      b_min_val = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      b_min_flag = true;

      iarg += 2;
    } else if (strcmp(arg[iarg], "beta") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for 'beta' in fix skmd");

      beta = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (beta < 0.0)
        error->all(FLERR, "Illegal fix skmd command: beta must be >= 0");

      iarg += 2;
    } else if (strcmp(arg[iarg], "buffer") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for 'buffer' in fix skmd");

      buffer_steps = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      if (buffer_steps < 0)
        error->all(FLERR, "Illegal fix skmd command: buffer must be >= 0");

      iarg += 2;
    } else {
      error->all(FLERR, "Unknown keyword '{}' in fix skmd", arg[iarg]);
    }
  }

  // may set ncoeff, dataset_size, dataset_N and fill qjs/fjs/jacs
  read_files(arg[5]);

  if (b_adaptive) {
    // median pairwise dataset distance, else fallback
    double b_local = 1.0;
    double b_fallback = 1.0;

    if (dataset_size >= 2 && ncoeff > 0) {
      if (comm->me == 0) {
        b_local = median_pairwise_distance((const double *const *)qjs, dataset_size, ncoeff);
        if (!(b_local > 0.0)) b_local = b_fallback;
      }
      MPI_Bcast(&b_local, 1, MPI_DOUBLE, 0, world);
    } else {
      b_local = b_fallback;
    }

    // optional bounds stop adaptive b running away (seen on Si(111): b -> ~27)
    if (b_max_flag) b_local = std::min(b_local, b_max_val);
    if (b_min_flag) b_local = std::max(b_local, b_min_val);

    b = b_local;
  } else {
    b = b_input;
  }

  if (comm->me == 0) {
    if (b_adaptive)
      utils::logmesg(lmp, "fix skmd: setting b adaptively to {}\n", b);
    else
      utils::logmesg(lmp, "fix skmd: setting b from command line to {}\n", b);
  }

  modify->addstep_compute_all(1);

  scalar_flag = 1;
  vector_flag = 1;
  extscalar = 0;
  extvector = 0;
  global_freq = 1;
  size_vector = 4;
  fnorm2 = 0.0;
}

double FixSKMD::compute_scalar()
{
  return fnorm2;
}

double FixSKMD::compute_vector(int n)
{
  if (n == 0) return fnorm2;
  if (n == 1) return b;
  if (n == 2) return triggered ? 1.0 : 0.0;
  // mean fnorm2 over the current run (less noisy than the last step)
  if (n == 3) return fnorm2_count > 0 ? fnorm2_sum / fnorm2_count : fnorm2;
  return 0.0;
}

FixSKMD::~FixSKMD()
{
  // a Kokkos functor copy (copymode=1) must not free the original's memory
  if (copymode) return;

  if (allocated) {
    memory->destroy(q1);
    memory->destroy(deriv_vec);
    memory->destroy(Fskmdx);
  }

  memory->destroy(qjs);
  memory->destroy(fjs);
  memory->destroy(jacs);

  delete[] compute_q_id;
  delete[] compute_jac_id;
  delete[] dataset_filename;

}

void FixSKMD::allocate()
{
  // empty dataset: lock dataset_N to the current atom count
  bigint Nnow = atom->natoms;
  if (dataset_N == 0) dataset_N = static_cast<int>(Nnow);

  // a different atom count makes the dataset forces/jacs inconsistent
  if (static_cast<bigint>(dataset_N) != Nnow) {
    error->all(FLERR,
               "SKMD dataset expects N={} atoms but current system has N={} atoms",
               dataset_N, Nnow);
  }

  const int nmax = atom->nmax;
  const bigint N = Nnow;

  memory->create(Fskmdx, 3 * N, "fixskmd:Fskmdx");
  memory->create(q1, ncoeff, "fixskmd:q1");
  memory->create(deriv_vec, ncoeff, "fixskmd:deriv_vec");

  allocated = true;
}

void FixSKMD::init()
{
  compute_q_ptr = modify->get_compute_by_id(compute_q_id);
  compute_jac_ptr = modify->get_compute_by_id(compute_jac_id);

  if (!compute_q_ptr)
    error->all(FLERR, "Compute ID {} does not exist", compute_q_id);

  if (!compute_jac_ptr)
    error->all(FLERR, "Compute ID {} does not exist", compute_jac_id);

  // reset per `run`: a reused fix gets a fresh buffer window (paper: buffer
  // steps since the last acquisition)
  run_start_step = update->ntimestep;
  triggered = false;
  fnorm2_sum = 0.0;
  fnorm2_count = 0;
}

int FixSKMD::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  mask |= POST_RUN;
  return mask;
}

void FixSKMD::post_force(int vflag)
{
  tagint *tag = atom->tag;
  double **f = atom->f;
  const int nlocal = atom->nlocal;

  v_init(vflag);
  if (!q_computed) return;

  // NOTE: the fix group is ignored; the bias goes on every local atom (see
  // README). tag[] is the 1-based global atom id.
  for (int i = 0; i < nlocal; i++) {
    const int id0 = static_cast<int>(tag[i]) - 1; // zero-based global
    const bigint base = 3 * static_cast<bigint>(id0);
    f[i][0] += Fskmdx[base + 0];
    f[i][1] += Fskmdx[base + 1];
    f[i][2] += Fskmdx[base + 2];
  }
}

double FixSKMD::k(const double *q, const double *q_prime) const
{
  return skmd_kernel(q, q_prime, ncoeff, a, b);
}

void FixSKMD::compute_k_deriv(const double *q, const double *q_prime)
{
  skmd_kernel_deriv(q, q_prime, ncoeff, a, b, deriv_vec);
}

void FixSKMD::end_of_step()
{
  const bigint N = atom->natoms;
  if (static_cast<bigint>(dataset_N) != N) {
    error->all(FLERR,
               "SKMD dataset expects N={} atoms but current system has N={} atoms",
               dataset_N, N);
  }

  q_computed = true;

  const int nlocal = atom->nlocal;

  // Invoke per-atom compute if needed
  modify->clearstep_compute();
  if (!(compute_q_ptr->invoked_flag & Compute::INVOKED_PERATOM)) {
    compute_q_ptr->compute_peratom();
    compute_q_ptr->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  modify->addstep_compute(update->ntimestep + 1);

  double **q_atom = compute_q_ptr->array_atom;

  const int ncols = compute_q_ptr->size_peratom_cols;
  if (q_atom == nullptr)
    error->all(FLERR, "Compute {} did not provide per-atom array data (array_atom is null)", compute_q_id);

  if (ncoeff == 0){
    ncoeff = ncols;
  }else{
    if (ncols != ncoeff) error->all(FLERR, "Compute {} returned {} per-atom cols but expected ncoeff={}",
                compute_q_id, ncols, ncoeff);
  }

  // now we know ncoeff for certain we can allocate arrays.
  if (!allocated) allocate();

  // Build global descriptor by summing local atoms then Allreduce
  for (int c = 0; c < ncoeff; ++c) q1[c] = 0.0;

  for (int i = 0; i < nlocal; ++i)
    for (int c = 0; c < ncoeff; ++c)
      q1[c] += q_atom[i][c];

  MPI_Allreduce(MPI_IN_PLACE, q1, ncoeff, MPI_DOUBLE, MPI_SUM, world);

  // SKMD force in Cartesian space: Fskmdx[0..3N-1]
  for (bigint r = 0; r < 3 * N; ++r) Fskmdx[r] = 0.0;

  const int J = dataset_size;
  if (J > 0) {
    const double invJ = 1.0 / static_cast<double>(J);

    for (int i = 0; i < J; ++i) {
      const double ki = k(qjs[i], q1);

      // attractive: beta * F(x_i) * k
      for (int aidx = 0; aidx < dataset_N; ++aidx) {
        const bigint base = 3 * static_cast<bigint>(aidx);
        Fskmdx[base + 0] += invJ * (beta * fjs[i][aidx][0] * ki);
        Fskmdx[base + 1] += invJ * (beta * fjs[i][aidx][1] * ki);
        Fskmdx[base + 2] += invJ * (beta * fjs[i][aidx][2] * ki);
      }

      // repulsive: grad_{x_i} k = (dq_i/dx_i) * grad_{q_i} k
      compute_k_deriv(qjs[i], q1); // deriv_vec[c] = d/d q_i[c] k(q_i, q1)

      for (bigint r = 0; r < 3 * N; ++r) {
        double acc = 0.0;
        for (int c = 0; c < ncoeff; ++c){
          acc += jacs[i][r][c] * deriv_vec[c];
        }
        Fskmdx[r] += invJ * acc;
      }
    }
  }

  double sumsq = 0.0;
  for (bigint r = 0; r < 3 * N; ++r) {
    sumsq += Fskmdx[r] * Fskmdx[r];
  }
  // NB: despite the name, fnorm2 is the Euclidean norm ||F_SKMD|| (not squared)
  fnorm2 = sqrt(sumsq);
  fnorm2_sum += fnorm2;
  fnorm2_count += 1;

  if (comm->me == 0) {
    const std::string base(dataset_filename);
    std::ofstream out(base + "_stats.txt", std::ios::app);
    if (out.is_open()) {
      out << update->ntimestep << " " << b << " " << fnorm2 << "\n";
    } else {
      error->warning(FLERR, "Could not open _stats.txt file for appending");
    }
  }

  if (thresh_flag && (update->ntimestep - run_start_step) >= buffer_steps) {
    if (fnorm2 < thresh) {
      if (comm->me == 0) {
        utils::logmesg(lmp,
          "Fix {} threshold met on step {}: {} < {}\n",
          id, update->ntimestep, fnorm2, thresh);
      }
      timer->force_timeout();
      triggered = true;
    }
  }
}

void FixSKMD::post_run() {
  if (triggered) timer->reset_timeout();
}

// Currently unused (the Python driver writes the dataset); kept for in-LAMMPS acquisition.
void FixSKMD::add_current_point_to_dataset()
{
  if (!allocated)
    error->all(FLERR, "Attempting to add data before arrays have been allocated", compute_q_id);

  const bigint N = atom->natoms;
  if (dataset_N == 0) dataset_N = static_cast<int>(N);
  if (static_cast<bigint>(dataset_N) != N) {
    error->all(FLERR,
               "Cannot append dataset point: dataset expects N={} atoms but current N={}",
               dataset_N, N);
  }

  // allocate dataset if it's not yet been allocated
  if (!dataset_allocated) {
    dataset_allocated = true;
    memory->destroy(qjs);
    memory->destroy(fjs);
    memory->destroy(jacs);

    memory->create(qjs, alloc_rows, ncoeff, "FixSKMD:qjs");
    memory->create(fjs, alloc_rows, dataset_N, 3, "FixSKMD:fjs");
    memory->create(jacs, alloc_rows, 3 * dataset_N, ncoeff, "FixSKMD:jacs");
  }

  // grow dataset arrays if needed
  dataset_size += 1;
  if (dataset_size > alloc_rows) {
    alloc_rows = 2 * dataset_size;
    memory->grow(qjs, alloc_rows, ncoeff, "FixSKMD:qjs");
    memory->grow(fjs, alloc_rows, dataset_N, 3, "FixSKMD:fjs");
    memory->grow(jacs, alloc_rows, 3 * dataset_N, ncoeff, "FixSKMD:jacs");
  }
  const int idx = dataset_size - 1;

  // store q: global descriptor q1 from end_of_step
  for (int c = 0; c < ncoeff; ++c) qjs[idx][c] = q1[c];

  // per-atom Jacobian compute (e.g. compute symmetrix/maced/atom)
  modify->clearstep_compute();
  if (!(compute_jac_ptr->invoked_flag & Compute::INVOKED_PERATOM)) {
    compute_jac_ptr->compute_peratom();
    compute_jac_ptr->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  modify->addstep_compute(update->ntimestep + 1);

  double **jac_atom = compute_jac_ptr->array_atom;
  const int jac_cols = compute_jac_ptr->size_peratom_cols;

  if (jac_atom == nullptr)
    error->all(FLERR, "Compute {} did not provide per-atom array data (array_atom is null)", compute_jac_id);

  // columns: (x,y,z) sub-blocks of ncoeff, per type block
  if (jac_cols % (3 * ncoeff) != 0)
    error->all(FLERR,
               "Compute {} returned size_peratom_cols={} but expected a multiple of 3*ncoeff={} (snad/atom-like layout)",
               compute_jac_id, jac_cols, 3 * ncoeff);

  const int ntype_blocks = jac_cols / (3 * ncoeff);

  // assemble global Jacobian (3N x ncoeff), then Allreduce
  std::vector<double> jacbuf(static_cast<size_t>(3 * dataset_N) * ncoeff, 0.0);

  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; ++i) {
    const int id0 = static_cast<int>(tag[i]) - 1;  // 0-based global atom id
    const int rowbase = 3 * id0;

    // For each Cartesian component d, sum across type blocks
    for (int d = 0; d < 3; ++d) {
      const int global_row = rowbase + d;
      double *dst = &jacbuf[static_cast<size_t>(global_row) * ncoeff];

      for (int c = 0; c < ncoeff; ++c) {
        double acc = 0.0;
        for (int tb = 0; tb < ntype_blocks; ++tb) {
          const int col = tb * (3 * ncoeff) + d * ncoeff + c;
          acc += jac_atom[i][col];
        }
        dst[c] = acc;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, jacbuf.data(),
                static_cast<int>(jacbuf.size()), MPI_DOUBLE, MPI_SUM, world);

  for (int r = 0; r < 3 * dataset_N; ++r) {
    const double *src = &jacbuf[static_cast<size_t>(r) * ncoeff];
    for (int c = 0; c < ncoeff; ++c) jacs[idx][r][c] = src[c];
  }

  // store physical forces f - Fskmdx in global ordering
  std::vector<double> fbuf(static_cast<size_t>(3 * dataset_N), 0.0);
  double **f = atom->f;

  for (int i = 0; i < nlocal; ++i) {
    const int id0 = static_cast<int>(tag[i]) - 1;
    const bigint base = 3 * static_cast<bigint>(id0);
    fbuf[static_cast<size_t>(base + 0)] = f[i][0] - Fskmdx[base + 0];
    fbuf[static_cast<size_t>(base + 1)] = f[i][1] - Fskmdx[base + 1];
    fbuf[static_cast<size_t>(base + 2)] = f[i][2] - Fskmdx[base + 2];
  }

  MPI_Allreduce(MPI_IN_PLACE, fbuf.data(),
                static_cast<int>(fbuf.size()), MPI_DOUBLE, MPI_SUM, world);

  for (int aidx = 0; aidx < dataset_N; ++aidx) {
    fjs[idx][aidx][0] = fbuf[static_cast<size_t>(3 * aidx + 0)];
    fjs[idx][aidx][1] = fbuf[static_cast<size_t>(3 * aidx + 1)];
    fjs[idx][aidx][2] = fbuf[static_cast<size_t>(3 * aidx + 2)];
  }

  // append to dataset files on rank 0
  if (comm->me == 0) {
    const std::string base(dataset_filename);

    // qs
    {
      std::fstream fs(base + "_qs.skdat", std::ios::in | std::ios::out);
      int npts = 0, cols = 0;
      fs >> npts >> cols;
      fs.seekp(0, std::ios::end);

      for (int c = 0; c < ncoeff; c++) fs << qjs[idx][c] << " ";
      fs << "\n";

      fs.seekp(0);
      fs << (npts + 1) << " " << ncoeff << "\n";
    }

    // fs
    {
      std::fstream fs(base + "_fs.skdat", std::ios::in | std::ios::out);
      int npts = 0, ncols2 = 0;
      fs >> npts >> ncols2;
      fs.seekp(0, std::ios::end);

      for (int aidx = 0; aidx < dataset_N; aidx++)
        for (int d = 0; d < 3; d++)
          fs << fjs[idx][aidx][d] << " ";
      fs << "\n";

      fs.seekp(0);
      fs << (npts + 1) << " " << dataset_N << "\n";
    }

    // jacs
    {
      std::fstream fs(base + "_jacs.skdat", std::ios::in | std::ios::out);
      int npts = 0, rows = 0, cols = 0;
      fs >> npts >> rows >> cols;
      fs.seekp(0, std::ios::end);

      for (int r = 0; r < 3 * dataset_N; r++)
        for (int c = 0; c < ncoeff; c++)
          fs << jacs[idx][r][c] << " ";
      fs << "\n";

      fs.seekp(0);
      fs << (npts + 1) << " " << (3 * dataset_N) << " " << ncoeff << "\n";
    }
  }

  MPI_Barrier(world);
}
// -----------------------------------------------------------------------------

void FixSKMD::read_files(char *dataset_base)
{
  // read dataset files, creating empty ones if missing
  const std::string base(dataset_base);
  const std::string qs_file = base + "_qs.skdat";
  const std::string fs_file = base + "_fs.skdat";
  const std::string jacs_file = base + "_jacs.skdat";

  int qs_exists = 0, fs_exists = 0, jacs_exists = 0;
  if (comm->me == 0) {
    qs_exists = std::ifstream(qs_file).good();
    fs_exists = std::ifstream(fs_file).good();
    jacs_exists = std::ifstream(jacs_file).good();
  }
  MPI_Bcast(&qs_exists, 1, MPI_INT, 0, world);
  MPI_Bcast(&fs_exists, 1, MPI_INT, 0, world);
  MPI_Bcast(&jacs_exists, 1, MPI_INT, 0, world);

  if (comm->me == 0) {
    if (!qs_exists) ensure_file_exists(qs_file, "0 0");
    if (!fs_exists) ensure_file_exists(fs_file, "0 0");
    if (!jacs_exists) ensure_file_exists(jacs_file, "0 0 0");
  }
  MPI_Barrier(world);

  int n_qs = 0, qs_cols = 0;
  int n_fs = 0, fs_cols = 0;
  int n_jacs = 0, jacs_rows = 0, jacs_cols = 0;

  if (comm->me == 0) {
    { std::ifstream ifs(qs_file);   ifs >> n_qs >> qs_cols; }
    { std::ifstream ifs(fs_file);   ifs >> n_fs >> fs_cols; }
    { std::ifstream ifs(jacs_file); ifs >> n_jacs >> jacs_rows >> jacs_cols; }
  }

  MPI_Bcast(&n_qs, 1, MPI_INT, 0, world);
  MPI_Bcast(&qs_cols, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_fs, 1, MPI_INT, 0, world);
  MPI_Bcast(&fs_cols, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_jacs, 1, MPI_INT, 0, world);
  MPI_Bcast(&jacs_rows, 1, MPI_INT, 0, world);
  MPI_Bcast(&jacs_cols, 1, MPI_INT, 0, world);

  const bool any_nonempty = (n_qs > 0) || (n_fs > 0) || (n_jacs > 0);
  dataset_size = n_qs;

  if (any_nonempty) {
    if (!(n_qs == n_fs && n_qs == n_jacs))
      error->all(FLERR, "Dataset files have inconsistent number of datapoints");

    if (qs_cols != jacs_cols)
      error->all(FLERR, "qs column count does not match jacs column count");

    ncoeff = qs_cols;

    if (jacs_rows % 3 != 0)
      error->all(FLERR, "jacs row count not divisible by 3");

    dataset_N = jacs_rows / 3;

    if (fs_cols != dataset_N)
      error->all(FLERR, "fs column count does not match N inferred from jacs");
  }

  alloc_rows = 2 * dataset_size;
  if (alloc_rows < 1) alloc_rows = 1;
  std::vector<double> buf_qs, buf_fs, buf_jacs;

  if (dataset_size > 0) {
    // every rank holds a full copy of the dataset (filled by the broadcast below)
    dataset_allocated = true;
    memory->destroy(qjs);
    memory->destroy(fjs);
    memory->destroy(jacs);
    memory->create(qjs, alloc_rows, ncoeff, "FixSKMD:qjs");
    memory->create(fjs, alloc_rows, dataset_N, 3, "FixSKMD:fjs");
    memory->create(jacs, alloc_rows, 3 * dataset_N, ncoeff, "FixSKMD:jacs");
  }

  if (comm->me == 0 && dataset_size > 0) {
    // read existing dataset into temporary buffers on rank 0 then broadcast
    buf_qs.resize(static_cast<size_t>(dataset_size) * ncoeff);
    buf_fs.resize(static_cast<size_t>(dataset_size) * dataset_N * 3);
    buf_jacs.resize(static_cast<size_t>(dataset_size) * 3 * dataset_N * ncoeff);

    // qs
    {
      std::ifstream ifs(qs_file);
      int dummy_n, dummy_c;
      ifs >> dummy_n >> dummy_c;
      for (int i = 0; i < dataset_size; ++i)
        for (int c = 0; c < ncoeff; ++c)
          ifs >> buf_qs[static_cast<size_t>(i) * ncoeff + c];
    }

    // fs
    {
      std::ifstream ifs(fs_file);
      int dummy_n, dummy_N;
      ifs >> dummy_n >> dummy_N;
      for (int i = 0; i < dataset_size; ++i)
        for (int aidx = 0; aidx < dataset_N; ++aidx)
          for (int d = 0; d < 3; ++d)
            ifs >> buf_fs[(static_cast<size_t>(i) * dataset_N + aidx) * 3 + d];
    }

    // jacs
    {
      std::ifstream ifs(jacs_file);
      int dummy_n, dummy_r, dummy_c;
      ifs >> dummy_n >> dummy_r >> dummy_c;
      for (int i = 0; i < dataset_size; ++i)
        for (int r = 0; r < 3 * dataset_N; ++r)
          for (int c = 0; c < ncoeff; ++c)
            ifs >> buf_jacs[(static_cast<size_t>(i) * (3 * dataset_N) + r) * ncoeff + c];
    }
  }

  if (dataset_size > 0) {
    if (comm->me != 0) {
      buf_qs.resize(static_cast<size_t>(dataset_size) * ncoeff);
      buf_fs.resize(static_cast<size_t>(dataset_size) * dataset_N * 3);
      buf_jacs.resize(static_cast<size_t>(dataset_size) * 3 * dataset_N * ncoeff);
    }

    MPI_Bcast(buf_qs.data(), dataset_size * ncoeff, MPI_DOUBLE, 0, world);
    MPI_Bcast(buf_fs.data(), dataset_size * dataset_N * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(buf_jacs.data(), dataset_size * 3 * dataset_N * ncoeff, MPI_DOUBLE, 0, world);

    // copy into LAMMPS arrays
    for (int i = 0; i < dataset_size; ++i) {
      for (int c = 0; c < ncoeff; ++c)
        qjs[i][c] = buf_qs[static_cast<size_t>(i) * ncoeff + c];

      for (int aidx = 0; aidx < dataset_N; ++aidx)
        for (int d = 0; d < 3; ++d)
          fjs[i][aidx][d] = buf_fs[(static_cast<size_t>(i) * dataset_N + aidx) * 3 + d];

      for (int r = 0; r < 3 * dataset_N; ++r)
        for (int c = 0; c < ncoeff; ++c)
          jacs[i][r][c] =
              buf_jacs[(static_cast<size_t>(i) * (3 * dataset_N) + r) * ncoeff + c];
    }
  }
}