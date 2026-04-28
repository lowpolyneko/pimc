#!/usr/bin/env python3
"""Generate synthetic binary inputs for benchmarking GPPotential.

The production GPPotential currently reads testdata.dat and proddata.dat from
the process working directory. These files are benchmark fixtures only; they do
not represent a physically meaningful GP model.
"""

import argparse
import math
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--points", type=int, default=512)
    args = parser.parse_args()

    if args.points <= 0:
        raise SystemExit("--points must be positive")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "testdata.dat").open("wb") as handle:
        for i in range(args.points):
            t = i / max(args.points - 1, 1)
            x = -0.8 + 1.6 * t
            y = 0.45 * math.sin(2.0 * math.pi * t)
            z = 0.25 + 0.5 * math.cos(2.0 * math.pi * t)
            flag = 1.0 if (i % 2) else 0.0
            handle.write(struct.pack("dddd", x, y, z, flag))

    with (out / "proddata.dat").open("wb") as handle:
        for i in range(args.points):
            coeff = 1.0e-4 * math.sin(0.017 * i) * math.cos(0.031 * i)
            handle.write(struct.pack("d", coeff))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
