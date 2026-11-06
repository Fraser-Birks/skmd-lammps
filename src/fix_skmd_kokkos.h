/* -*- c++ -*- ----------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

// fix skmd/kk is NOT a device port: the maths stays on the host. It only narrows
// the Kokkos datamask (F_MASK|TAG_MASK) to avoid syncing atom->x every step.

#ifdef FIX_CLASS
// clang-format off
FixStyle(skmd/kk,FixSKMDKokkos<LMPDeviceType>);
FixStyle(skmd/kk/device,FixSKMDKokkos<LMPDeviceType>);
FixStyle(skmd/kk/host,FixSKMDKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_SKMD_KOKKOS_H
#define LMP_FIX_SKMD_KOKKOS_H

#include "fix_skmd.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixSKMDKokkos : public FixSKMD {
 public:
  FixSKMDKokkos(class LAMMPS *, int, char **);
  ~FixSKMDKokkos() override = default;

  void post_force(int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
