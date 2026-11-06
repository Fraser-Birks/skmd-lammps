#!/usr/bin/env python3
"""Check the SKMD bias force computed by `fix skmd` against an independent
NumPy implementation of

    F_SKMD(x) = (1/J) sum_i [ beta k(q_i, q(x)) F_i + Jac_i^T grad_{q_i} k(q_i, q(x)) ],
    k(q, q') = a exp(-|q - q'|^2 / (2 b^2)),

for a random dataset (random descriptors q_i, forces F_i and Jacobians Jac_i),
on a 12-atom system with `pair_style zero` and no integrator, so the per-atom
forces LAMMPS reports are exactly the bias. The current-configuration
descriptor is q(x) = sum over atoms of (x, y, z) from `compute property/atom`.
Checked: fixed b, `adaptive` b (median pairwise dataset distance), and a
non-default `beta`.

Run it under several MPI rank counts: the result must not depend on the
domain decomposition.

Usage:
  check_force.py --lmp "mpirun -np 4 /path/lmp" [--plugin fixskmdplugin.so] [-- extra lmp args]
"""
import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N_ATOMS, NCOEFF, J = 12, 3, 5
A, B_FIXED = 2.0, 4.0
BETA = 0.37  # non-default attractive-term weight; must move the force well beyond TOL
TOL = 1e-9  # relative, against the largest force component


def write_inputs(work: Path, rng: np.random.Generator):
    box = 10.0
    pos = rng.uniform(0.5, box - 0.5, size=(N_ATOMS, 3))
    with open(work / "atoms.data", "w") as fh:
        fh.write(f"skmd force test\n\n{N_ATOMS} atoms\n1 atom types\n\n"
                 f"0 {box} xlo xhi\n0 {box} ylo yhi\n0 {box} zlo zhi\n\n"
                 "Masses\n\n1 1.0\n\nAtoms # atomic\n\n")
        for i, p in enumerate(pos, start=1):
            fh.write(f"{i} 1 {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n")

    q_now = pos.sum(axis=0)
    # dataset descriptors scattered around q_now so kernel values are O(1)
    qs = q_now + rng.normal(scale=3.0, size=(J, NCOEFF))
    fs = rng.normal(size=(J, N_ATOMS, 3))
    jacs = rng.normal(size=(J, 3 * N_ATOMS, NCOEFF))

    fmt = lambda arr: " ".join(f"{v:.17e}" for v in np.ravel(arr))
    (work / "ds_qs.skdat").write_text(f"{J} {NCOEFF}\n" + "\n".join(fmt(q) for q in qs) + "\n")
    (work / "ds_fs.skdat").write_text(f"{J} {N_ATOMS}\n" + "\n".join(fmt(f) for f in fs) + "\n")
    (work / "ds_jacs.skdat").write_text(
        f"{J} {3 * N_ATOMS} {NCOEFF}\n" + "\n".join(fmt(jc) for jc in jacs) + "\n")
    return pos, qs, fs, jacs


def reference_force(q_now, qs, fs, jacs, a, b, beta=1.0):
    force = np.zeros(3 * N_ATOMS)
    for q_i, f_i, jac_i in zip(qs, fs, jacs):
        d = q_i - q_now
        k = a * np.exp(-d @ d / (2 * b * b))
        grad_qi_k = -d / (b * b) * k
        force += beta * k * f_i.ravel() + jac_i @ grad_qi_k
    return (force / len(qs)).reshape(N_ATOMS, 3)


def median_pairwise(qs):
    return float(np.median([np.linalg.norm(qs[i] - qs[j])
                            for i in range(len(qs)) for j in range(i + 1, len(qs))]))


LMP_INPUT = """\
{plugin}
units           metal
atom_style      atomic
atom_modify     map yes sort 0 0.0
boundary        p p p
read_data       atoms.data
pair_style      zero 3.0
pair_coeff      * *
compute         q   all property/atom x y z
compute         jac all property/atom fx fy fz
fix             skmd all skmd q jac ds {a} {b}{kw}
thermo_style    custom step f_skmd f_skmd[2]
thermo          1
thermo_modify   format float %.17e
dump            d all custom 2 forces_{tag}.dump id fx fy fz
dump_modify     d sort id format float %.17e
# no integrator: positions stay fixed. The bias computed at the end of step 1
# is applied in post_force of step 2 and appears in the step-2 dump.
run             2
print           "SKMD_CHECK b=$(f_skmd[2]:%.17e) fnorm=$(f_skmd:%.17e)"
"""


def run_case(args, work, b_arg, tag, kw=""):
    plugin = f"plugin load {Path(args.plugin).resolve()}" if args.plugin else ""
    (work / f"in.{tag}").write_text(LMP_INPUT.format(plugin=plugin, a=A, b=b_arg, tag=tag, kw=kw))
    for suffix in ("qs", "fs", "jacs"):  # fix skmd never writes these, but be safe
        assert (work / f"ds_{suffix}.skdat").exists()
    cmd = shlex.split(args.lmp) + ["-in", f"in.{tag}", "-log", f"log.{tag}"] + args.extra
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        sys.exit(f"FAIL [{tag}]: LAMMPS exited with code {proc.returncode}")
    line = next(l for l in (work / f"log.{tag}").read_text().splitlines() if l.startswith("SKMD_CHECK"))
    b_lmp = float(line.split("b=")[1].split()[0])
    fnorm_lmp = float(line.split("fnorm=")[1].split()[0])
    lines = (work / f"forces_{tag}.dump").read_text().splitlines()
    last = max(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    f_lmp = np.array([[float(v) for v in l.split()[1:4]] for l in lines[last + 1:last + 1 + N_ATOMS]])
    return b_lmp, fnorm_lmp, f_lmp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmp", required=True, help='LAMMPS command, e.g. "mpirun -np 4 lmp"')
    p.add_argument("--plugin", default=None, help="fixskmdplugin.so to `plugin load` (omit for in-tree)")
    p.add_argument("extra", nargs="*", help="extra LAMMPS args after --, e.g. -k on -sf kk")
    args = p.parse_args()

    work = Path(tempfile.mkdtemp(prefix="skmd_force_"))
    pos, qs, fs, jacs = write_inputs(work, np.random.default_rng(20261104))
    q_now = pos.sum(axis=0)

    for tag, b_arg, b_ref, beta in (("fixed", str(B_FIXED), B_FIXED, 1.0),
                                    ("adaptive", "adaptive", median_pairwise(qs), 1.0),
                                    ("beta", str(B_FIXED), B_FIXED, BETA)):
        kw = f" beta {beta!r}" if beta != 1.0 else ""
        b_lmp, fnorm_lmp, f_lmp = run_case(args, work, b_arg, tag, kw)
        f_ref = reference_force(q_now, qs, fs, jacs, A, b_ref, beta)
        scale = np.abs(f_ref).max()
        err = np.abs(f_lmp - f_ref).max() / scale
        print(f"[{tag}] b={b_lmp:.6g} (ref {b_ref:.6g})  max|F|={scale:.4g}  "
              f"max rel err={err:.2e}  |F| lmp={fnorm_lmp:.6g} ref={np.linalg.norm(f_ref):.6g}")
        if abs(b_lmp - b_ref) > 1e-12 * b_ref:
            sys.exit(f"FAIL [{tag}]: b={b_lmp} but reference b={b_ref}")
        if not err < TOL:
            sys.exit(f"FAIL [{tag}]: bias force differs from reference (max rel err {err:.3e})")
        if abs(fnorm_lmp - np.linalg.norm(f_ref)) > TOL * np.linalg.norm(f_ref):
            sys.exit(f"FAIL [{tag}]: reported |F_SKMD| {fnorm_lmp} != reference {np.linalg.norm(f_ref)}")
    print("FORCE CHECK PASSED")


if __name__ == "__main__":
    main()
