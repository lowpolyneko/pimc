#!/usr/bin/env python3
"""Generate synthetic binary inputs for benchmarking the generic GP potential.

The binary training file stores rows of x,y,z,K^{-1}Y. These files are
benchmark fixtures only; they do not represent a physically meaningful model.
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

    training_file = out / "gp_training.dat"
    with training_file.open("wb") as handle:
        for i in range(args.points):
            t = i / max(args.points - 1, 1)
            x = -0.8 + 1.6 * t
            y = 0.45 * math.sin(2.0 * math.pi * t)
            z = 0.25 + 0.5 * math.cos(2.0 * math.pi * t)
            coeff = 1.0e-4 * math.sin(0.017 * i) * math.cos(0.031 * i)
            handle.write(struct.pack("dddd", x, y, z, coeff))

    with (out / "gp_input.ini").open("w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "kernel.type = matern",
                    "kernel.meanType = constant",
                    "kernel.maternNu = 2.5",
                    "kernel.ell = 0.78638807",
                    "kernel.ell = 2.17270815",
                    "kernel.ell = 0.77220716",
                    "kernel.mean = 22.0553313",
                    "kernel.sigma2 = 814.69663559",
                    f"kernel.numTrainingPoints = {args.points}",
                    f"kernel.trainingFileName = {training_file}",
                    "data.normOffset = 0.0",
                    "data.normOffset = 0.0",
                    "data.normOffset = -0.09459459",
                    "data.normScale = 7.0",
                    "data.normScale = 3.5",
                    "data.normScale = 7.09459459",
                    "data.standardMean = 8.64215696",
                    "data.standardStd = 104.91133484",
                    "",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
