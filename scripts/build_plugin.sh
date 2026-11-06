#!/usr/bin/env bash
# Build the SKMD LAMMPS plugin (fixskmdplugin.so, plain `fix skmd` only).
#
# Usage:
#   LAMMPS_DIR=/path/to/lammps scripts/build_plugin.sh [extra cmake args]
#
# Requirements: an MPI build of LAMMPS with -D PKG_PLUGIN=yes (and, for use
# from Python, -D BUILD_SHARED_LIBS=yes -D PKG_PYTHON=yes); C++17 compiler;
# CMake >= 3.16; MPI. The plugin must be compiled against the same LAMMPS
# source tree (headers) as the LAMMPS binary/library that loads it.
#
# For the Kokkos variant (fix skmd/kk) use the in-tree route instead:
# src/install.sh + a KOKKOS build of LAMMPS (see build_lammps_kokkos_cuda.sh).
#
# Add -DSKMD_TESTING=ON to also build the standalone kernel unit tests.
#
# Part of skmd-lammps. GPL-2.0, see LICENSE.
set -euo pipefail

: "${LAMMPS_DIR:?set LAMMPS_DIR to the LAMMPS source checkout (containing src/)}"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-${REPO_DIR}/build}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

cmake -S "${REPO_DIR}/cmake" -B "${BUILD_DIR}" \
  -D LAMMPS_SOURCE_DIR="${LAMMPS_DIR}/src" \
  -D CMAKE_BUILD_TYPE=Release \
  "$@"

cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo "Built: ${BUILD_DIR}/fixskmdplugin.so"
