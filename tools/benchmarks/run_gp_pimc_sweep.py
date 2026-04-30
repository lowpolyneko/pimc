#!/usr/bin/env python3
"""Build, run end-to-end PIMC GP potential benchmarks, and write tables."""

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
TIME_RE = re.compile(r"^real\s+([0-9.eE+-]+)$")
KV_RE = re.compile(r"([A-Za-z_]+)=([0-9.eE+-]+)")


def int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
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

    build_cmd = ["cmake", "--build", str(build_dir), f"-j{args.jobs}", "--target", "pimc.e"]
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


def parse_output(output: str) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {
        "real_seconds": None,
        "batched_seconds": None,
        "batched_calls": None,
        "batched_positions": None,
        "local_scalar_calls": None,
        "local_range_calls": None,
        "local_list_calls": None,
        "bare_scalar_calls": None,
    }
    for line in output.splitlines():
        line = line.strip()
        match = TIME_RE.match(line)
        if match:
            values["real_seconds"] = float(match.group(1))
            continue
        if line.startswith("batched_external_potential"):
            kv = dict(KV_RE.findall(line))
            if "seconds" in kv:
                values["batched_seconds"] = float(kv["seconds"])
            if "calls" in kv:
                values["batched_calls"] = int(float(kv["calls"]))
            if "positions" in kv:
                values["batched_positions"] = int(float(kv["positions"]))
        if line.startswith("action_call_stats"):
            kv = dict(KV_RE.findall(line))
            for key in ("local_scalar_calls", "local_range_calls", "local_list_calls", "bare_scalar_calls"):
                if key in kv:
                    values[key] = int(float(kv[key]))
    return values


def update_args(update: str) -> list[str]:
    return [] if update == "default" else ["--update", update]


def run_case(exe: Path, mode: str, gp: int, particles: int, slices: int, update: str, data_dir: Path, args):
    env = os.environ.copy()
    env["PIMC_ACTION_CALL_STATS"] = "1"
    env["PIMC_BATCHED_POTENTIAL_STATS"] = "1"
    if mode.startswith("gpu"):
        env["ONEAPI_DEVICE_SELECTOR"] = args.gpu_selector
    if mode == "gpu_scalar":
        env["PIMC_DISABLE_BATCHED_POTENTIAL"] = "1"
    else:
        env.pop("PIMC_DISABLE_BATCHED_POTENTIAL", None)

    pimc_cmd = [
        str(exe),
        "--no_save_state",
        "-T",
        "1.0",
        "-N",
        str(particles),
        "-L",
        "10.0",
        "-P",
        str(slices),
        "--action",
        "gsf",
        "-X",
        "gp_he_benzene",
        "-I",
        "free",
        "--number_eq_steps",
        str(args.eq_steps),
        "--bin_size",
        str(args.bin_size),
        "--number_bins_stored",
        str(args.bins),
        "--gp_input",
        str(data_dir / "gp_input.ini"),
    ] + update_args(update)
    result = run(["/usr/bin/time", "-p"] + pimc_cmd, env=env, timeout=args.timeout, dry_run=args.dry_run)
    print(result.stdout, end="")
    parsed = parse_output(result.stdout)
    return [
        gp,
        particles,
        slices,
        update,
        mode,
        parsed["real_seconds"],
        parsed["batched_seconds"],
        parsed["batched_calls"],
        parsed["batched_positions"],
        parsed["local_scalar_calls"],
        parsed["local_range_calls"],
        parsed["local_list_calls"],
        parsed["bare_scalar_calls"],
        result.returncode,
    ]


def write_outputs(rows: list[list], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gp_pimc_sweep.csv"
    md_path = output_dir / "gp_pimc_sweep.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gp_points",
            "particles",
            "slices",
            "update",
            "mode",
            "real_seconds",
            "batched_seconds",
            "batched_calls",
            "batched_positions",
            "local_scalar_calls",
            "local_range_calls",
            "local_list_calls",
            "bare_scalar_calls",
            "returncode",
        ])
        writer.writerows(rows)

    by_case: dict[tuple[int, int, int, str], dict[str, list]] = {}
    for row in rows:
        by_case.setdefault((row[0], row[1], row[2], row[3]), {})[row[4]] = row

    def fmt(value):
        return "ERR" if value is None else (str(value) if isinstance(value, int) else f"{value:.6g}")

    def speedup(num, den):
        return "ERR" if num is None or den in (None, 0) else f"{num / den:.2f}x"

    lines = [
        "# GP Potential End-to-End PIMC Sweep",
        "",
        "Times are wall-clock `real` seconds from `/usr/bin/time -p`.",
        "",
        "| GP pts | N | P | update | CPU | GPU scalar | GPU batched | CPU/GPU batched | GPU scalar/GPU batched | GPU batched positions | local list calls | bare scalar calls |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(by_case):
        modes = by_case[key]
        cpu = modes.get("cpu", [None] * 14)[5]
        gpu_scalar = modes.get("gpu_scalar", [None] * 14)[5]
        gpu_batched_row = modes.get("gpu_batched")
        gpu_batched = gpu_batched_row[5] if gpu_batched_row else None
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {fmt(cpu)} | "
            f"{fmt(gpu_scalar)} | {fmt(gpu_batched)} | {speedup(cpu, gpu_batched)} | "
            f"{speedup(gpu_scalar, gpu_batched)} | "
            f"{fmt(gpu_batched_row[8] if gpu_batched_row else None)} | "
            f"{fmt(gpu_batched_row[11] if gpu_batched_row else None)} | "
            f"{fmt(gpu_batched_row[12] if gpu_batched_row else None)} |"
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
    parser.add_argument("--gp-points", type=int_list, default=int_list("16384"))
    parser.add_argument("--particles", type=int_list, default=int_list("16,64"))
    parser.add_argument("--slices", type=int_list, default=int_list("16,64"))
    parser.add_argument("--updates", type=str_list, default=str_list("default,bisection"))
    parser.add_argument("--bin-size", type=int, default=64)
    parser.add_argument("--eq-steps", type=int, default=0)
    parser.add_argument("--bins", type=int, default=1)
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
        modes.append(("cpu", args.cpu_build_dir / "pimc.e"))
    if not args.skip_gpu_scalar:
        modes.append(("gpu_scalar", args.gpu_build_dir / "pimc.e"))
    if not args.skip_gpu_batched:
        modes.append(("gpu_batched", args.gpu_build_dir / "pimc.e"))

    rows = []
    for gp in args.gp_points:
        data_dir = ensure_gp_data(gp, args.data_root, args.dry_run)
        for particles in args.particles:
            for slices in args.slices:
                for update in args.updates:
                    for mode, exe in modes:
                        print(
                            f"CASE mode={mode} gp={gp} N={particles} P={slices} update={update}",
                            flush=True,
                        )
                        rows.append(run_case(exe, mode, gp, particles, slices, update, data_dir, args))

    if not args.dry_run:
        write_outputs(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
