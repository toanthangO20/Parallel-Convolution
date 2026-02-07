#!/usr/bin/env python3
"""
Run seq, mpi, and mpi_omp convolution and compare output files byte-by-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def resolve_input_path(raw: str, repo_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    choices = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, repo_root / candidate]
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Input file not found: {raw}")


def resolve_local_exe(raw: str, repo_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        bases = [candidate]
    else:
        bases = [Path.cwd() / candidate, repo_root / candidate]

    seen = set()
    for base in bases:
        checks = [base]
        if base.suffix.lower() != ".exe":
            checks.append(Path(str(base) + ".exe"))
        for path in checks:
            norm = str(path.resolve(strict=False))
            if norm in seen:
                continue
            seen.add(norm)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"Executable not found: {raw}")


def resolve_mpiexec(raw: str, repo_root: Path) -> str:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.is_absolute():
            checks = [candidate]
        else:
            checks = [Path.cwd() / candidate, repo_root / candidate]
        for path in checks:
            if path.is_file():
                return str(path.resolve())
        raise FileNotFoundError(f"mpiexec not found: {raw}")

    found = shutil.which(raw)
    if found:
        return found
    raise FileNotFoundError(f"Command not found in PATH: {raw}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compare_files(lhs: Path, rhs: Path) -> tuple[bool, str]:
    lhs_size = lhs.stat().st_size
    rhs_size = rhs.stat().st_size
    if lhs_size != rhs_size:
        return False, f"size mismatch ({lhs_size} vs {rhs_size} bytes)"

    offset = 0
    with lhs.open("rb") as f1, rhs.open("rb") as f2:
        while True:
            b1 = f1.read(1024 * 1024)
            b2 = f2.read(1024 * 1024)
            if not b1 and not b2:
                return True, "identical"
            if b1 != b2:
                limit = min(len(b1), len(b2))
                for i in range(limit):
                    if b1[i] != b2[i]:
                        return False, f"first mismatch at byte {offset + i}: {b1[i]} != {b2[i]}"
                return False, f"mismatch near byte {offset + limit}"
            offset += len(b1)


def run_and_collect(
    name: str,
    cmd: list[str],
    cwd: Path,
    out_file: Path,
    expected_size: int,
    snapshot_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[str, int]:
    print(f"[run] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout.rstrip())
        print(result.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    if not out_file.is_file():
        raise RuntimeError(f"{name} did not generate output file: {out_file.name}")

    output_size = out_file.stat().st_size
    if output_size != expected_size:
        raise RuntimeError(
            f"{name} output size mismatch: {output_size} bytes (expected {expected_size})"
        )

    shutil.copy2(out_file, snapshot_path)
    return sha256_file(snapshot_path), output_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seq/mpi/mpi_omp and compare blur outputs byte-by-byte."
    )
    parser.add_argument("--input", required=True, help="Path to input raw image")
    parser.add_argument("--width", type=int, required=True, help="Image width")
    parser.add_argument("--height", type=int, required=True, help="Image height")
    parser.add_argument("--loops", type=int, default=20, help="Convolution loop count (default: 20)")
    parser.add_argument("--mode", choices=["grey", "rgb"], default="grey", help="Image mode (default: grey)")
    parser.add_argument("--np", type=int, default=4, help="MPI process count (default: 4)")
    parser.add_argument("--omp-threads", type=int, default=4, help="OMP_NUM_THREADS for mpi_omp (default: 4)")
    parser.add_argument("--mpiexec", default="mpiexec", help="mpiexec command/path (default: mpiexec)")
    parser.add_argument("--seq-exe", default="seq/seq_conv", help="Path to seq executable")
    parser.add_argument("--mpi-exe", default="mpi/mpi_conv", help="Path to mpi executable")
    parser.add_argument("--mpi-omp-exe", default="mpi_omp/mpi_omp_conv", help="Path to mpi_omp executable")
    parser.add_argument(
        "--save-outdir",
        default=None,
        help="Optional directory to save blur_seq/blur_mpi/blur_mpi_omp snapshots",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary working directory for debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.width <= 0 or args.height <= 0:
        print("width and height must be positive integers", file=sys.stderr)
        return 1
    if args.loops < 0:
        print("loops must be >= 0", file=sys.stderr)
        return 1
    if args.np <= 0:
        print("np must be >= 1", file=sys.stderr)
        return 1
    if args.omp_threads <= 0:
        print("omp-threads must be >= 1", file=sys.stderr)
        return 1

    try:
        input_path = resolve_input_path(args.input, repo_root)
        seq_exe = resolve_local_exe(args.seq_exe, repo_root)
        mpi_exe = resolve_local_exe(args.mpi_exe, repo_root)
        mpi_omp_exe = resolve_local_exe(args.mpi_omp_exe, repo_root)
        mpiexec = resolve_mpiexec(args.mpiexec, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    bytes_per_pixel = 1 if args.mode == "grey" else 3
    expected_size = args.width * args.height * bytes_per_pixel
    input_size = input_path.stat().st_size
    if input_size != expected_size:
        print(
            f"Input size mismatch: {input_size} bytes (expected {expected_size})",
            file=sys.stderr,
        )
        return 1

    save_dir = None
    if args.save_outdir:
        save_dir = Path(args.save_outdir).expanduser()
        if not save_dir.is_absolute():
            save_dir = Path.cwd() / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="compare_outputs_"))
    print(f"[info] temp dir: {temp_dir}")
    if args.keep_temp:
        print("[info] keep-temp enabled; directory will not be removed")

    statuses: dict[str, tuple[str, int]] = {}
    try:
        temp_input = temp_dir / input_path.name
        shutil.copy2(input_path, temp_input)
        blur_name = f"blur_{temp_input.name}"
        blur_path = temp_dir / blur_name

        snapshots = {
            "seq": temp_dir / "blur_seq.raw",
            "mpi": temp_dir / "blur_mpi.raw",
            "mpi_omp": temp_dir / "blur_mpi_omp.raw",
        }

        if blur_path.exists():
            blur_path.unlink()
        statuses["seq"] = run_and_collect(
            "seq",
            [
                str(seq_exe),
                temp_input.name,
                str(args.width),
                str(args.height),
                str(args.loops),
                args.mode,
            ],
            temp_dir,
            blur_path,
            expected_size,
            snapshots["seq"],
        )

        if blur_path.exists():
            blur_path.unlink()
        statuses["mpi"] = run_and_collect(
            "mpi",
            [
                mpiexec,
                "-n",
                str(args.np),
                str(mpi_exe),
                temp_input.name,
                str(args.width),
                str(args.height),
                str(args.loops),
                args.mode,
            ],
            temp_dir,
            blur_path,
            expected_size,
            snapshots["mpi"],
        )

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        if blur_path.exists():
            blur_path.unlink()
        statuses["mpi_omp"] = run_and_collect(
            "mpi_omp",
            [
                mpiexec,
                "-n",
                str(args.np),
                str(mpi_omp_exe),
                temp_input.name,
                str(args.width),
                str(args.height),
                str(args.loops),
                args.mode,
            ],
            temp_dir,
            blur_path,
            expected_size,
            snapshots["mpi_omp"],
            env=env,
        )

        print("")
        print("Output hashes:")
        print(f"  seq     : {statuses['seq'][0]} ({statuses['seq'][1]} bytes)")
        print(f"  mpi     : {statuses['mpi'][0]} ({statuses['mpi'][1]} bytes)")
        print(f"  mpi_omp : {statuses['mpi_omp'][0]} ({statuses['mpi_omp'][1]} bytes)")

        ok_all = True
        for name in ("mpi", "mpi_omp"):
            same, detail = compare_files(snapshots["seq"], snapshots[name])
            if same:
                print(f"[match] seq vs {name}: identical")
            else:
                print(f"[mismatch] seq vs {name}: {detail}")
                ok_all = False

        if save_dir is not None:
            for name, path in snapshots.items():
                target = save_dir / f"blur_{name}.raw"
                shutil.copy2(path, target)
            print(f"[info] saved snapshots to: {save_dir}")

        if ok_all:
            print("[pass] All outputs are identical")
            return 0
        print("[fail] Outputs are different")
        return 2

    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
