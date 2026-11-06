/* ----------------------------------------------------------------------
   skmd-lammps: Stein Kernelized Molecular Dynamics (SKMD) for LAMMPS
   https://github.com/Fraser-Birks/skmd-lammps

   Copyright (C) 2025-2026 Fraser Birks

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2, as
   published by the Free Software Foundation. It is distributed WITHOUT
   ANY WARRANTY; see the LICENSE file for details.
------------------------------------------------------------------------- */

#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "fix_skmd.h"

using namespace LAMMPS_NS;

static Fix *fixskmdcreator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixSKMD(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "skmd";
  plugin.info = "SKMD fix style v0.1.0";
  plugin.author = "Fraser Birks (https://github.com/Fraser-Birks/skmd-lammps)";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fixskmdcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}