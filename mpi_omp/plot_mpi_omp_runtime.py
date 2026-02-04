from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent

WIDTH = 1920
HEIGHTS = [630, 1260, 2520, 5040]
PS = [1, 2, 4, 9, 16, 25]


def format_comma(value, decimals=2):
    fmt = f"{value:.{decimals}f}"
    return fmt.replace('.', ',')


def load_times(csv_path):
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                image_type = row["image_type"].strip()
                width = int(row["width"])
                height = int(row["height"])
                p = int(row["p"])
            except (KeyError, ValueError):
                continue
            val = row.get("runtime_seconds", "").strip()
            if not val or val.lower() in ("none", "nan", "--"):
                rt = None
            else:
                try:
                    rt = float(val)
                except ValueError:
                    rt = None
            data[(image_type, width, height, p)] = rt
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot MPI+OpenMP runtime comparison")
    parser.add_argument("--csv", default=str(BASE_DIR / "table2_mpi_omp_times.csv"), help="CSV file from benchmark_table2_mpi_omp.py")
    parser.add_argument("--outdir", default=str(BASE_DIR), help="Output directory")
    args = parser.parse_args()

    data = load_times(args.csv)

    cases = [
        ("grey", 630, "grey\n1920*\n630 (x/4)"),
        ("grey", 1260, "grey\n1920*\n1260 (x/2)"),
        ("grey", 2520, "grey\n1920*\n2520 (x)"),
        ("grey", 5040, "grey\n1920*\n5040 (2x)"),
        ("rgb", 630, "rgb\n1920*\n630 (x/4)"),
        ("rgb", 1260, "rgb\n1920*\n1260 (x/2)"),
        ("rgb", 2520, "rgb\n1920*\n2520 (x)"),
        ("rgb", 5040, "rgb\n1920*\n5040 (2x)"),
    ]

    labels = [c[2] for c in cases]
    x = list(range(len(labels)))

    missing = []
    series = {}
    for p in PS:
        y = []
        for image_type, height, _ in cases:
            rt = data.get((image_type, WIDTH, height, p))
            if rt is None:
                missing.append((image_type, WIDTH, height, p))
                y.append(float("nan"))
            else:
                y.append(rt)
        series[p] = y

    if missing:
        print(f"WARNING: missing {len(missing)} entries in CSV (showing up to 5)", file=sys.stderr)
        for item in missing[:5]:
            print(f"  missing: {item}", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(11, 6))

    for p in PS:
        y = series[p]
        ax.plot(x, y, linewidth=2, label=format_comma(p, 2))
        ax.fill_between(x, 0, y, alpha=0.2)

    ax.set_title("Processes: 1, 2, 4, 9, 16, 25")
    ax.set_xlabel("Image Size / Processes")
    ax.set_ylabel("Runtime (s)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.grid(axis="y", alpha=0.3)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: format_comma(v, 2)))

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.subplots_adjust(right=0.78)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "mpi_omp_runtime.png"
    pdf_path = out_dir / "mpi_omp_runtime.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(str(png_path))
    print(str(pdf_path))


if __name__ == "__main__":
    main()
