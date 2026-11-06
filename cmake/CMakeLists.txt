# ----------------------------------------------------------------------
# skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
# https://github.com/Fraser-Birks/skmd-lammps
#
# CMake build for the SKMD LAMMPS plugin (fixskmdplugin.so).
# Adapted from the plugin example (examples/plugins) in the LAMMPS source.
# Copyright (C) 2025-2026 Axel Kohlmeyer, Fraser Birks
#
# Distributed under the GNU General Public License, version 2, the same
# license as LAMMPS. See the LICENSE file for details.
# ----------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# enforce out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed. You must create and use a build directory. "
    "Please remove CMakeCache.txt and CMakeFiles first.")
endif()

project(skmd_lammps_plugin VERSION 1.0 LANGUAGES CXX)

# when this file is included as subdirectory in the LAMMPS build, many settings are directly imported
if(LAMMPS_DIR)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
else() 
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR} CACHE PATH "Location of LAMMPS headers")
  if(NOT LAMMPS_HEADER_DIR)
    message(FATAL_ERROR "Must set LAMMPS_HEADER_DIR")
  endif()
  # by default, install into $HOME/.local (not /usr/local),
  # so that no root access (and sudo) is needed
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
  # ugly hacks for MSVC which by default always reports an old C++ standard in the __cplusplus macro
  # and prints lots of pointless warnings about "unsafe" functions
  if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      add_compile_options(/Zc:__cplusplus)
      add_compile_options(/wd4244)
      add_compile_options(/wd4267)
      add_compile_options(/wd4250)
      add_compile_options(/EHsc)
    endif()
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  endif()
endif()

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Need -restrict with Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(CheckIncludeFileCXX)
if(NOT LAMMPS_DIR)
  include(LAMMPSInterfaceCXX)
endif()

##########################
# building the plugin
# plain `fix skmd` only; fix skmd/kk needs the in-tree KOKKOS build (src/install.sh)
set(SRC_DIR ../src)

add_library(fixskmdplugin MODULE ${SRC_DIR}/skmd_plugin.cpp ${SRC_DIR}/fix_skmd.cpp)
target_link_libraries(fixskmdplugin PRIVATE lammps)

set_target_properties(fixskmdplugin PROPERTIES PREFIX "" SUFFIX ".so")

# MacOS seems to need this
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(fixskmdplugin
    PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
# tell CMake to export all symbols to a .dll on Windows with special case for MinGW cross-compilers
  set_target_properties(fixskmdplugin
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(fixskmdplugin
      PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(fixskmdplugin PROPERTIES
    LINK_FLAGS "-rdynamic")
endif()

add_custom_target(plugins ALL ${CMAKE_COMMAND} -E echo "Building Plugins"
  DEPENDS fixskmdplugin)

##########################
# optional kernel unit tests (no LAMMPS linkage required)
option(SKMD_TESTING "Build standalone kernel unit tests" OFF)
if(SKMD_TESTING)
  add_executable(test_kernel ../tests/test_kernel.cpp)
  target_include_directories(test_kernel PRIVATE ${SRC_DIR})
  set_target_properties(test_kernel PROPERTIES CXX_STANDARD 17)
  message(STATUS "SKMD kernel tests enabled — binary: test_kernel")
endif()
