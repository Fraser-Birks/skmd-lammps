#!/usr/bin/env bash
# Example: CPU build of LAMMPS + symmetrix (MACE) + SKMD, as used for the
# SKMD paper's CPU runs. Adapt modules/compilers to your machine.
#
# Usage:
#   LAMMPS_DIR=/path/to/lammps \
#   SYMMETRIX_DIR=/path/to/symmetrix-public \
#   scripts/build_lammps_cpu.sh
#
# Environment variables:
#   LAMMPS_DIR      LAMMPS source checkout (paper: commit d3b14f6c49,
#                   patch_4Jul2026-277; symmetrix needs >= 10 Sep 2025)
#   SYMMETRIX_DIR   checkout of https://github.com/Fraser-Birks/symmetrix-public,
#                   tag skmd-v1.0 (paper: commit 10171c9), cloned
#                   with --recursive (libsymmetrix has submodules)
#   SKMD_IN_TREE    1 (default, as in the paper): compile fix skmd into LAMMPS
#                   via src/install.sh. 0: leave it out and use the plugin
#                   (scripts/build_plugin.sh + `plugin load`) instead.
#   INSTALL_PREFIX  install prefix (default: $VIRTUAL_ENV, else $HOME/.local)
#   BUILD_DIR       build directory (default: $LAMMPS_DIR/build-skmd-cpu)
#   JOBS            parallel build jobs (default: nproc)
#
# Package set: PYTHON (the SKMD driver is Python), BROWNIAN (fix brownian,
# for the overdamped runs), PLUGIN (to load fixskmdplugin.so if not in-tree).
# Tested on the University of Warwick "Avon" cluster with GCC 13.2,
# OpenMPI 4.1.6, CMake 3.27.
#
# Part of skmd-lammps. GPL-2.0, see LICENSE.
set -euo pipefail

: "${LAMMPS_DIR:?set LAMMPS_DIR to the LAMMPS source checkout}"
: "${SYMMETRIX_DIR:?set SYMMETRIX_DIR to the symmetrix-public checkout (tag skmd-v1.0)}"
SKMD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SKMD_IN_TREE=${SKMD_IN_TREE:-1}
INSTALL_PREFIX=${INSTALL_PREFIX:-${VIRTUAL_ENV:-$HOME/.local}}
BUILD_DIR=${BUILD_DIR:-${LAMMPS_DIR}/build-skmd-cpu}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

# 1. symmetrix (guarded: its install.sh appends to LAMMPS' cmake/CMakeLists.txt)
if ! grep -q "libsymmetrix" "${LAMMPS_DIR}/cmake/CMakeLists.txt"; then
  (cd "${SYMMETRIX_DIR}/pair_symmetrix" && ./install.sh "${LAMMPS_DIR}")
else
  echo "symmetrix already registered in ${LAMMPS_DIR}/cmake/CMakeLists.txt; skipping its install.sh"
fi

# 2. SKMD (in-tree)
if [ "${SKMD_IN_TREE}" = "1" ]; then
  "${SKMD_DIR}/src/install.sh" "${LAMMPS_DIR}"
fi

# 3. configure + build + install LAMMPS and its Python module
cmake -S "${LAMMPS_DIR}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D BUILD_MPI=yes \
  -D BUILD_SHARED_LIBS=yes \
  -D PKG_PYTHON=yes \
  -D PKG_BROWNIAN=yes \
  -D PKG_PLUGIN=yes \
  -D SYMMETRIX_KOKKOS=no \
  -D SPHERICART_ENABLE_CUDA=OFF

cmake --build "${BUILD_DIR}" -j "${JOBS}"
cmake --install "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target install-python
