#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import subprocess
import statistics
import random
from pathlib import Path
import shutil

WIDTH = 1920
HEIGHTS = [630, 1260, 2520, 5040]
PS = [1, 2, 4, 9, 16, 25]
IMAGE_TYPES = ["grey", "rgb"]
LOOPS = 20
REPEATS = 3
SEED = 123


def generate_data_file(path: Path, size: int) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    data = bytearray(rng.getrandbits(8) for _ in range(size))
    path.write_bytes(data)


def parse_runtime(output: str):
    text = output.strip().split()
    if not text:
        return None
    try:
        return float(text[-1])
    except ValueError:
        return None


def format_number(val):
    if val is None:
        return "--"
    s = f"{val:.2f}"
    return s.replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Table 1 MPI runtimes")
    default_exe = "./mpi_conv.exe" if os.name == "nt" else "./mpi_conv"
    parser.add_argument("--exe", default=default_exe, help="Path to mpi_conv binary")
    parser.add_argument("--mpiexec", default="mpiexec", help="mpiexec path")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="Repeats per case")
    args = parser.parse_args()

    exe_path = args.exe
    mpiexec = args.mpiexec
    repeats = args.repeats

    results = []
    data_dir = Path("data")
    error_log = Path("table1_mpi_errors.log")
    error_log.write_text("", encoding="ascii")

    exe_path_obj = Path(exe_path)
    if not exe_path_obj.exists():
        candidates = []
        if os.name == "nt" and not exe_path.endswith(".exe"):
            candidates.append(Path(exe_path + ".exe"))
        candidates.append(Path("mpi") / exe_path_obj.name)
        if os.name == "nt" and not exe_path_obj.name.endswith(".exe"):
            candidates.append(Path("mpi") / (exe_path_obj.name + ".exe"))
        for cand in candidates:
            if cand.exists():
                exe_path_obj = cand
                break
        if not exe_path_obj.exists():
            print(f"WARNING: mpi binary not found at: {exe_path}", file=sys.stderr)
    exe_path = str(exe_path_obj.resolve())

    mpiexec_path = shutil.which(mpiexec) if Path(mpiexec).name == mpiexec else None
    if mpiexec_path:
        mpiexec = mpiexec_path
    elif not Path(mpiexec).exists():
        print(f"WARNING: mpiexec not found: {mpiexec}", file=sys.stderr)

    for image_type in IMAGE_TYPES:
        for height in HEIGHTS:
            if image_type == "grey":
                size = WIDTH * height
            else:
                size = WIDTH * height * 3
            filename = f"{image_type}_{WIDTH}x{height}.bin"
            data_path = data_dir / filename
            generate_data_file(data_path, size)

            for p in PS:
                runtimes = []
                for _ in range(repeats):
                    cmd = [mpiexec, "-n", str(p), exe_path, str(data_path), str(WIDTH), str(height), str(LOOPS), image_type]
                    try:
                        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    except Exception as e:
                        error_log.write_text(
                            error_log.read_text(encoding="ascii")
                            + f"EXCEPTION: {' '.join(cmd)}\n"
                            + f"error: {e}\n\n",
                            encoding="ascii",
                        )
                        runtimes.append(None)
                        continue

                    if proc.returncode != 0:
                        error_log.write_text(
                            error_log.read_text(encoding="ascii")
                            + f"FAIL: {' '.join(cmd)}\n"
                            + f"stdout: {proc.stdout}\n"
                            + f"stderr: {proc.stderr}\n\n",
                            encoding="ascii",
                        )
                        runtimes.append(None)
                        continue

                    rt = parse_runtime(proc.stdout)
                    if rt is None and proc.stderr:
                        rt = parse_runtime(proc.stderr)
                    if rt is None:
                        error_log.write_text(
                            error_log.read_text(encoding="ascii")
                            + f"PARSE_FAIL: {' '.join(cmd)}\n"
                            + f"stdout: {proc.stdout}\n"
                            + f"stderr: {proc.stderr}\n\n",
                            encoding="ascii",
                        )
                        runtimes.append(None)
                    else:
                        runtimes.append(rt)

                vals = [v for v in runtimes if v is not None]
                if vals:
                    median_rt = statistics.median(vals)
                else:
                    median_rt = None
                results.append((image_type, WIDTH, height, p, median_rt))

    csv_path = Path("table1_mpi_times.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_type", "width", "height", "p", "runtime_seconds"])
        for row in results:
            writer.writerow(row)

    # Build LaTeX table
    def size_label(h):
        if h == 630:
            return "(x/4)"
        if h == 1260:
            return "(x/2)"
        if h == 2520:
            return "(x)"
        if h == 5040:
            return "(2x)"
        return ""

    lines = []
    lines.append("\\begin{tabular}{|l|r|r|r|r|r|r|}\\hline")
    lines.append("Image size & 1 & 2 & 4 & 9 & 16 & 25 \\\\ \\\\hline")

    for image_type in IMAGE_TYPES:
        for height in HEIGHTS:
            label = f"{image_type} {WIDTH}$\\times${height} {size_label(height)}"
            row = [label]
            for p in PS:
                rt = None
                for r in results:
                    if r[0] == image_type and r[2] == height and r[3] == p:
                        rt = r[4]
                        break
                row.append(format_number(rt))
            lines.append("{} & {} \\\\".format(row[0], " & ".join(row[1:])))
        lines.append("\\hline")

    lines.append("\\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
