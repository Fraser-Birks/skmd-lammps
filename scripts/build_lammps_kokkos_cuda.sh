#!/usr/bin/env bash
# Example: KOKKOS/CUDA build of LAMMPS + symmetrix (MACE, GPU) + SKMD
# (fix skmd and fix skmd/kk compiled in-tree), as used for the SKMD paper's
# GPU runs. Run it on a GPU node, not a login node. Adapt modules to your machine.
#
# Usage:
#   LAMMPS_DIR=/path/to/lammps \
#   SYMMETRIX_DIR=/path/to/symmetrix-public \
#   scripts/build_lammps_kokkos_cuda.sh
#
# Environment variables:
#   LAMMPS_DIR       LAMMPS source checkout (paper: commit d3b14f6c49,
#                    patch_4Jul2026-277; symmetrix needs >= 10 Sep 2025)
#   SYMMETRIX_DIR    checkout of https://github.com/Fraser-Birks/symmetrix-public,
#                    tag skmd-v1.0 (paper: commit 10171c9), cloned
#                    with --recursive
#   KOKKOS_GPU_ARCH  optional Kokkos GPU arch, e.g. TURING75 (Quadro RTX 6000),
#                    AMPERE80 (A100), HOPPER90 (H100). Empty (default): let
#                    Kokkos auto-detect the GPU of the build node.
#   CXX_FLAGS        extra compiler flags (default: "-march=native -ffast-math",
#                    as used for the paper)
#   INSTALL_PREFIX   install prefix (default: $VIRTUAL_ENV, else $HOME/.local)
#   BUILD_DIR        build directory (default: $LAMMPS_DIR/build-skmd-kokkos)
#   JOBS             parallel build jobs (default: nproc)
#
# fix skmd/kk is only available from an in-tree build like this one; the
# CMake plugin (fixskmdplugin.so) contains only the plain fix skmd.
#
# Tested on the University of Warwick "Avon" cluster: Quadro RTX 6000,
# GCC 13.2, OpenMPI 4.1.6, CUDA 12.6, CMake 3.27.
#
# Part of skmd-lammps. GPL-2.0, see LICENSE.
set -euo pipefail

: "${LAMMPS_DIR:?set LAMMPS_DIR to the LAMMPS source checkout}"
: "${SYMMETRIX_DIR:?set SYMMETRIX_DIR to the symmetrix-public checkout (tag skmd-v1.0)}"
SKMD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KOKKOS_GPU_ARCH=${KOKKOS_GPU_ARCH:-}
CXX_FLAGS=${CXX_FLAGS:--march=native -ffast-math}
INSTALL_PREFIX=${INSTALL_PREFIX:-${VIRTUAL_ENV:-$HOME/.local}}
BUILD_DIR=${BUILD_DIR:-${LAMMPS_DIR}/build-skmd-kokkos}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

# 1. symmetrix (guarded: its install.sh appends to LAMMPS' cmake/CMakeLists.txt)
if ! grep -q "libsymmetrix" "${LAMMPS_DIR}/cmake/CMakeLists.txt"; then
  (cd "${SYMMETRIX_DIR}/pair_symmetrix" && ./install.sh "${LAMMPS_DIR}")
else
  echo "symmetrix already registered in ${LAMMPS_DIR}/cmake/CMakeLists.txt; skipping its install.sh"
fi

# 2. SKMD in-tree (fix skmd + fix skmd/kk)
"${SKMD_DIR}/src/install.sh" "${LAMMPS_DIR}"

# 3. configure + build + install
arch_args=()
if [ -n "${KOKKOS_GPU_ARCH}" ]; then
  arch_args+=(-D "Kokkos_ARCH_${KOKKOS_GPU_ARCH}=ON")
fi

cmake -S "${LAMMPS_DIR}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -D CMAKE_CXX_COMPILER="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper" \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_OMP=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_NATIVE=ON \
  -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
  "${arch_args[@]}" \
  -D SYMMETRIX_KOKKOS=ON \
  -D SYMMETRIX_SPHERICART_CUDA=ON \
  -D PKG_PYTHON=ON \
  -D PKG_BROWNIAN=ON

cmake --build "${BUILD_DIR}" -j "${JOBS}"
cmake --install "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target install-python
