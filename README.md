# skmd-lammps

[![CI](https://github.com/Fraser-Birks/skmd-lammps/actions/workflows/ci.yml/badge.svg)](https://github.com/Fraser-Birks/skmd-lammps/actions/workflows/ci.yml)

`fix skmd`: Stein Kernelized Molecular Dynamics for LAMMPS, as a plugin or
in-tree (with a KOKKOS wrapper, `fix skmd/kk`). The paper's active learning
pipelines are in [skmd-examples](https://github.com/Fraser-Birks/skmd-examples).

## Requirements

* LAMMPS with MPI (paper: commit `d3b14f6c49`, 4 Jul 2026); packages `BROWNIAN`,
  `PYTHON`, and `PLUGIN` or `KOKKOS` as needed.
* For MACE descriptors: [symmetrix-public](https://github.com/Fraser-Birks/symmetrix-public),
  tag `skmd-v1.0` (the paper used `10171c9`, before a periodic-image Jacobian
  fix), which provides `symmetrix/mace/atom` (descriptors) and `symmetrix/maced/atom` (Jacobian).
  Both need `atom_modify map yes`.

## Build

Plugin (`fix skmd` only; LAMMPS built with `-D PKG_PLUGIN=yes -D BUILD_SHARED_LIBS=yes`):

```bash
LAMMPS_DIR=/path/to/lammps scripts/build_plugin.sh     # -> build/fixskmdplugin.so
```

In-tree (needed for `fix skmd/kk`):

```bash
src/install.sh /path/to/lammps     # then build LAMMPS as usual
```

LAMMPS + symmetrix + SKMD in one go:

```bash
LAMMPS_DIR=/path/to/lammps SYMMETRIX_DIR=/path/to/symmetrix-public \
  scripts/build_lammps_kokkos_cuda.sh       # or build_lammps_cpu.sh
```

## Tests

```bash
cmake -S cmake -B build -D LAMMPS_SOURCE_DIR=/path/to/lammps/src -D SKMD_TESTING=ON
cmake --build build --target test_kernel && ./build/test_kernel
tests/smoke/run_smoke.sh /path/to/lmp build/fixskmdplugin.so
python3 tests/force/check_force.py --lmp /path/to/lmp --plugin build/fixskmdplugin.so
```

## Usage

```
fix ID group-ID skmd q-ID jac-ID dataset-base a b [keyword value ...]
```

| Argument | |
|---|---|
| `q-ID` | per-atom descriptor compute; q is its column sum |
| `jac-ID` | per-atom Jacobian compute (must exist) |
| `dataset-base` | prefix of the reference set files (below) |
| `a` | kernel amplitude |
| `b` | kernel bandwidth, or `adaptive` (median pairwise distance of the reference set) |

| Keyword | | Default |
|---|---|---|
| `thresh` ζ₀ | stop the run when ‖F_SKMD‖ < ζ₀ | off |
| `buffer` ℓ₀ | steps before `thresh` is tested | 0 |
| `beta` | weight of the attractive term (1/energy) | 1.0 |
| `b_min`, `b_max` | clamps on adaptive b | none |

The bias on the current configuration, with reference set {(q_j, F_j, J_j)}:

```
k(q_j, q) = a exp(-|q_j - q|² / (2 b²))
F_SKMD    = (1/J) Σ_j [ beta k(q_j, q) F_j + J_jᵀ ∇_{q_j} k(q_j, q) ]
```

It is applied with a one-step lag, to all atoms regardless of group.
`beta` = 1/(k_B T) gives Eq. (5b) of the paper; the paper's MACE runs used
`beta 1` (1/eV).

Output: `f_ID` = ‖F_SKMD‖; `f_ID[1..4]` = ‖F_SKMD‖, b, triggered (0/1), mean
‖F_SKMD‖ over the run.

Reference set (plain text, read once when the fix is created; atom IDs 1..N):

| File | Header | One line per point |
|---|---|---|
| `<base>_qs.skdat` | `npts ncoeff` | descriptor |
| `<base>_fs.skdat` | `npts N` | 3N forces |
| `<base>_jacs.skdat` | `npts 3N ncoeff` | Jacobian, row-major |

Example input: [`examples/in.skmd`](examples/in.skmd).

## Citation

```bibtex
@inproceedings{zou2026skmd,
  title     = {Stein Kernelized Molecular Dynamics for Active Learning of Interatomic Potentials},
  author    = {Zou, Joanna and Birks, Fraser and Foster, Dallas and Marzouk, Youssef},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

GPL-2.0
