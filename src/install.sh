#!/usr/bin/env bash
# Install fix skmd (and, if the KOKKOS package sources are present, fix skmd/kk)
# into a LAMMPS source tree by symlinking the sources, for an in-tree build.
#
#   Usage: src/install.sh /path/to/lammps
#
# Afterwards configure/build LAMMPS with CMake as usual. With -D PKG_KOKKOS=on
# the Kokkos wrapper in src/KOKKOS/ is picked up automatically (the KOKKOS
# package globs that directory). Do not also `plugin load` fixskmdplugin.so
# into a LAMMPS binary built this way: the style `skmd` would be defined twice.
#
# Part of skmd-lammps. GPL-2.0, see LICENSE.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage:    $0 path/to/lammps"
    exit 1
fi

lammps=$(cd "$1" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ ! -f "${lammps}/src/lammps.h" ]; then
    echo "Error: ${lammps} does not look like a LAMMPS source tree (no src/lammps.h)"
    exit 1
fi

# plain fix skmd
ln -sf "${here}/fix_skmd.h"    "${lammps}/src/fix_skmd.h"
ln -sf "${here}/fix_skmd.cpp"  "${lammps}/src/fix_skmd.cpp"
ln -sf "${here}/skmd_kernel.h" "${lammps}/src/skmd_kernel.h"

# Kokkos datamask wrapper (fix skmd/kk); lives in src/KOKKOS/ like any other
# KOKKOS-package file. Only compiled when LAMMPS is configured with PKG_KOKKOS=on.
if [ -d "${lammps}/src/KOKKOS" ]; then
    ln -sf "${here}/fix_skmd_kokkos.h"   "${lammps}/src/KOKKOS/fix_skmd_kokkos.h"
    ln -sf "${here}/fix_skmd_kokkos.cpp" "${lammps}/src/KOKKOS/fix_skmd_kokkos.cpp"
fi

echo "fix skmd installed into ${lammps}/src (symlinks to ${here})"
