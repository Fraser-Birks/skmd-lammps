"""Write a tiny synthetic SKMD dataset for the CI smoke test.

The smoke-test system has N = 32 atoms and uses `compute property/atom x y z`
as a stand-in descriptor, so ncoeff = 3 and the global descriptor is the sum of
atomic positions. Its Jacobian dq_c/dx_{i,d} is delta_{cd} for every atom.

Usage: python3 make_dataset.py <dataset_base>
"""
import sys

N, NCOEFF = 32, 3
base = sys.argv[1]
qs = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]   # median pairwise distance = 5
npts = len(qs)

with open(f"{base}_qs.skdat", "w") as fp:
    fp.write(f"{npts} {NCOEFF}\n")
    for q in qs:
        fp.write(" ".join(map(str, q)) + "\n")

with open(f"{base}_fs.skdat", "w") as fp:
    fp.write(f"{npts} {N}\n")
    for _ in qs:
        fp.write(" ".join(["0.1"] * (3 * N)) + "\n")

with open(f"{base}_jacs.skdat", "w") as fp:
    fp.write(f"{npts} {3 * N} {NCOEFF}\n")
    for _ in qs:
        row = []
        for _atom in range(N):
            for d in range(3):
                row += ["1.0" if c == d else "0.0" for c in range(NCOEFF)]
        fp.write(" ".join(row) + "\n")
