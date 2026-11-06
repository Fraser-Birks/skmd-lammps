/* ----------------------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

#include "fix_skmd_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos_base.h"

using namespace LAMMPS_NS;

template<class DeviceType>
FixSKMDKokkos<DeviceType>::FixSKMDKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixSKMD(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;

  // execution_space stays Host (as FixColvarsKokkos): Device would make ModifyKokkos
  // claim a Device modification alongside ours on Host, which DualView rejects.
  // post_force() only uses f and tag; end_of_step() reads only the compute output.
  datamask_read = F_MASK | TAG_MASK;
  datamask_modify = F_MASK;
}

template<class DeviceType>
void FixSKMDKokkos<DeviceType>::post_force(int vflag)
{
  // host loop, so sync/modify on Host; marking Device would silently drop the bias
  atomKK->sync(Host, datamask_read);
  FixSKMD::post_force(vflag);
  atomKK->modified(Host, datamask_modify);
}

namespace LAMMPS_NS {
template class FixSKMDKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSKMDKokkos<LMPHostType>;
#endif
}
