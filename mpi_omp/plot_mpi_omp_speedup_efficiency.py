from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

INCLUDE_P1 = False


def load_times(csv_path, image_type, width, height):
    times = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r_image = row["image_type"].strip()
                r_width = int(row["width"])
                r_height = int(row["height"])
                p = int(row["p"])
            except (KeyError, ValueError):
                continue
            if r_image != image_type or r_width != width or r_height != height:
                continue
            val = row.get("runtime_seconds", "").strip()
            if not val or val.lower() in ("none", "nan", "--"):
                rt = None
            else:
                try:
                    rt = float(val)
                except ValueError:
                    rt = None
            times[p] = rt
    return times


def compute_speedup(times, include_p1):
    t1 = times.get(1)
    if t1 is None:
        raise ValueError("Missing runtime for p=1")
    processes = sorted(times.keys())
    if not include_p1:
        processes = [p for p in processes if p != 1]
    speedup = {}
    for p in processes:
        tp = times.get(p)
        if tp is None:
            continue
        speedup[p] = t1 / tp
    return speedup


def compute_efficiency(speedup):
    return {p: s / p for p, s in speedup.items()}


def plot_line(x, y, title, y_label, y_lim, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, linewidth=3)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Processes")
    ax.set_ylabel(y_label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in x])

    ax.grid(axis="y", color="#d9d9d9", linewidth=1)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    border = plt.Rectangle(
        (0, 0),
        1,
        1,
        transform=fig.transFigure,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )
    fig.add_artist(border)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MPI+OpenMP speedup and efficiency")
    parser.add_argument("--csv", default=str(BASE_DIR / "table2_mpi_omp_times.csv"), help="CSV file from benchmark_table2_mpi_omp.py")
    parser.add_argument("--outdir", default=str(BASE_DIR), help="Output directory")
    parser.add_argument("--include-p1", action="store_true", help="Include p=1 on the x-axis")
    args = parser.parse_args()

    include_p1 = INCLUDE_P1 or args.include_p1

    times = load_times(args.csv, image_type="grey", width=1920, height=2520)
    if not times:
        print("ERROR: no matching entries for grey 1920x2520 in CSV", file=sys.stderr)
        raise SystemExit(1)

    speedup = compute_speedup(times, include_p1)
    efficiency = compute_efficiency(speedup)

    x = sorted(speedup.keys())
    y_speedup = [speedup[p] for p in x]
    y_efficiency = [efficiency[p] for p in x]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speedup_png = out_dir / "mpi_omp_speedup.png"
    speedup_pdf = out_dir / "mpi_omp_speedup.pdf"
    efficiency_png = out_dir / "mpi_omp_efficiency.png"
    efficiency_pdf = out_dir / "mpi_omp_efficiency.pdf"

    plot_line(
        x,
        y_speedup,
        "Grey 1920 * 2520",
        "Speedup",
        (0, 25),
        speedup_png,
        speedup_pdf,
    )

    plot_line(
        x,
        y_efficiency,
        "Grey 1920 * 2520",
        "Efficiency",
        (0, 1.2),
        efficiency_png,
        efficiency_pdf,
    )

    print(str(speedup_png))
    print(str(speedup_pdf))
    print(str(efficiency_png))
    print(str(efficiency_pdf))


if __name__ == "__main__":
    main()
