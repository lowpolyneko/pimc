#!/usr/bin/env python3
"""Build, generate fake GP data, run action benchmarks, and write tables."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPILER = "/opt/aurora/26.26.0/oneapi/compiler/latest/bin/icpx"
ACTION_RE = re.compile(r"^(action[UV]\([^)]+\)) calls=\d+ seconds=([0-9.eE+-]+)")


def int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def run(cmd: list[str], *, env=None, timeout=None, dry_run=False):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def configure_and_build(build_dir: Path, backend: str, args) -> None:
    if args.skip_build:
        return
    configure_cmd = [
        "cmake",
        "-S",
        str(ROOT),
        "-B",
        str(build_dir),
        f"-DCMAKE_CXX_COMPILER={args.compiler}",
        f"-DGPU_BACKEND={backend}",
        f"-DBOOST_ROOT={args.boost_root}",
        "-DBoost_NO_SYSTEM_PATHS=ON",
    ]
    result = run(configure_cmd, dry_run=args.dry_run)
    print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.returncode)

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        f"-j{args.jobs}",
        "--target",
        "potential_benchmark.e",
    ]
    result = run(build_cmd, dry_run=args.dry_run)
    print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.returncode)


def ensure_gp_data(points: int, data_root: Path, dry_run: bool) -> Path:
    out = data_root / f"points-{points}"
    if (out / "gp_input.ini").exists() and (out / "gp_training.dat").exists():
        return out
    cmd = [
        sys.executable,
        str(ROOT / "tools/benchmarks/generate_gp_fake_data.py"),
        "--output-dir",
        str(out),
        "--points",
        str(points),
    ]
    result = run(cmd, dry_run=dry_run)
    print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.returncode)
    return out


def parse_action_times(output: str) -> tuple[float | None, float | None]:
    action_v = None
    action_u = None
    for line in output.splitlines():
        match = ACTION_RE.match(line.strip())
        if not match:
            continue
        if match.group(1) == "actionV(slice)":
            action_v = float(match.group(2))
        elif match.group(1) == "actionU(range)":
            action_u = float(match.group(2))
    return action_v, action_u


def run_case(exe: Path, mode: str, gp: int, particles: int, slices: int, data_dir: Path, args):
    env = os.environ.copy()
    env["PIMC_ACTION_CALL_STATS"] = "1"
    env["PIMC_BATCHED_POTENTIAL_STATS"] = "1"
    if mode.startswith("gpu"):
        env["ONEAPI_DEVICE_SELECTOR"] = args.gpu_selector
    if mode == "gpu_scalar":
        env["PIMC_DISABLE_BATCHED_POTENTIAL"] = "1"
    else:
        env.pop("PIMC_DISABLE_BATCHED_POTENTIAL", None)

    cmd = [
        str(exe),
        "--benchmark-kind",
        "external",
        "--benchmark-potential",
        "gp_he_benzene",
        "--benchmark-method",
        "action",
        "--benchmark-iterations",
        str(args.iterations),
        "--action",
        "gsf",
        "-N",
        str(particles),
        "-P",
        str(slices),
        "-L",
        "10.0",
        "-T",
        "1.0",
        "--benchmark-min",
        "0.5",
        "--benchmark-max",
        "8.0",
        "--gp_input",
        str(data_dir / "gp_input.ini"),
    ]
    result = run(cmd, env=env, timeout=args.timeout, dry_run=args.dry_run)
    print(result.stdout, end="")
    action_v, action_u = parse_action_times(result.stdout)
    return [gp, particles, slices, mode, action_v, action_u, result.returncode]


def write_outputs(rows: list[list], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gp_action_sweep.csv"
    md_path = output_dir / "gp_action_sweep.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gp_points",
            "particles",
            "slices",
            "mode",
            "action_v_seconds",
            "action_u_seconds",
            "returncode",
        ])
        writer.writerows(rows)

    by_case: dict[tuple[int, int, int], dict[str, list]] = {}
    for row in rows:
        by_case.setdefault(tuple(row[:3]), {})[row[3]] = row

    def fmt(value):
        return "ERR" if value is None else f"{value:.6g}"

    def speedup(num, den):
        return "ERR" if num is None or den in (None, 0) else f"{num / den:.1f}x"

    lines = [
        "# GP Potential Action Sweep",
        "",
        "Times are total seconds for `actionU(range)` over the configured iteration count.",
        "",
        "| GP pts | N | P | CPU scalar | GPU scalar | GPU batched | CPU/GPU batched | GPU scalar/GPU batched |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(by_case):
        modes = by_case[key]
        cpu = modes.get("cpu", [None] * 6)[5]
        gpu_scalar = modes.get("gpu_scalar", [None] * 6)[5]
        gpu_batched = modes.get("gpu_batched", [None] * 6)[5]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {fmt(cpu)} | "
            f"{fmt(gpu_scalar)} | {fmt(gpu_batched)} | "
            f"{speedup(cpu, gpu_batched)} | {speedup(gpu_scalar, gpu_batched)} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-build-dir", type=Path, default=ROOT / "build-cpu-gp")
    parser.add_argument("--gpu-build-dir", type=Path, default=ROOT / "build-sycl-gp")
    parser.add_argument("--data-root", type=Path, default=ROOT / "gp_bench_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--gp-points", type=int_list, default=int_list("1024,4096,16384,32768"))
    parser.add_argument("--particles", type=int_list, default=int_list("16,64,256,1024"))
    parser.add_argument("--slices", type=int_list, default=int_list("16,64,256,1024"))
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--compiler", default=os.environ.get("CXX", DEFAULT_COMPILER))
    parser.add_argument("--boost-root", type=Path, default=Path("/tmp/local/boost/boost-v1.90.0"))
    parser.add_argument("--gpu-selector", default="level_zero:gpu")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu-scalar", action="store_true")
    parser.add_argument("--skip-gpu-batched", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.skip_cpu:
        configure_and_build(args.cpu_build_dir, "none", args)
    if not (args.skip_gpu_scalar and args.skip_gpu_batched):
        configure_and_build(args.gpu_build_dir, "sycl", args)

    modes = []
    if not args.skip_cpu:
        modes.append(("cpu", args.cpu_build_dir / "potential_benchmark.e"))
    if not args.skip_gpu_scalar:
        modes.append(("gpu_scalar", args.gpu_build_dir / "potential_benchmark.e"))
    if not args.skip_gpu_batched:
        modes.append(("gpu_batched", args.gpu_build_dir / "potential_benchmark.e"))

    rows = []
    for gp in args.gp_points:
        data_dir = ensure_gp_data(gp, args.data_root, args.dry_run)
        for particles in args.particles:
            for slices in args.slices:
                for mode, exe in modes:
                    print(f"CASE mode={mode} gp={gp} N={particles} P={slices}", flush=True)
                    rows.append(run_case(exe, mode, gp, particles, slices, data_dir, args))

    if not args.dry_run:
        write_outputs(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
