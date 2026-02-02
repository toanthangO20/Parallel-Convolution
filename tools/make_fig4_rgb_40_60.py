#!/usr/bin/env python3
# Requires Pillow. If missing, install: pip install pillow

import argparse
import subprocess
import sys
from pathlib import Path


def eprint(*args):
    print(*args, file=sys.stderr)


def load_raw(path: Path, width: int, height: int, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    expected = width * height * 3
    size = path.stat().st_size
    if size != expected:
        raise ValueError(
            f"{label} file size mismatch: {size} bytes (expected {expected})"
        )
    data = path.read_bytes()
    if len(data) != expected:
        raise ValueError(
            f"{label} read size mismatch: {len(data)} bytes (expected {expected})"
        )
    return data


def ensure_pillow():
    try:
        from PIL import Image  # type: ignore
    except Exception:
        eprint("Pillow is required. Install it with: pip install pillow")
        sys.exit(1)
    return Image


def run_seq_conv(exe: Path, input_name: str, width: int, height: int, loops: int, cwd: Path):
    cmd = [str(exe), input_name, str(width), str(height), str(loops), "rgb"]
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found or not runnable: {exe}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = "\n".join([s for s in [stdout, stderr] if s])
        msg = "Executable failed"
        if details:
            msg = f"{msg}:\n{details}"
        raise RuntimeError(msg) from exc


def main():
    parser = argparse.ArgumentParser(description="Generate RGB figure for two iteration counts.")
    parser.add_argument("--input", required=True, help="Path to input raw RGB image")
    parser.add_argument("--width", type=int, required=True, help="Image width")
    parser.add_argument("--height", type=int, required=True, help="Image height")
    parser.add_argument("--exe", required=True, help="Path to seq_conv executable")
    parser.add_argument("--outdir", default="Figures", help="Output directory for PNGs")
    parser.add_argument("--loops", nargs="+", type=int, default=[40, 60], help="Two iteration counts, e.g. --loops 40 60")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        eprint("width and height must be positive")
        return 1

    if len(args.loops) != 2:
        eprint("--loops must provide exactly two values (e.g. --loops 40 60)")
        return 1

    if any(l < 0 for l in args.loops):
        eprint("loops must be >= 0")
        return 1

    input_path = Path(args.input)
    exe_path = Path(args.exe)

    if not exe_path.is_file():
        eprint(f"Executable not found: {exe_path}")
        return 1

    input_dir = input_path.parent if input_path.parent != Path("") else Path(".")
    input_name = input_path.name

    Image = ensure_pillow()

    try:
        raw_input = load_raw(input_path, args.width, args.height, "Input")
    except Exception as exc:
        eprint(str(exc))
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images = []
    for loops in args.loops:
        if loops == 0:
            raw = raw_input
        else:
            try:
                run_seq_conv(exe_path, input_name, args.width, args.height, loops, input_dir)
            except Exception as exc:
                eprint(str(exc))
                return 1

            blur_path = input_dir / f"blur_{input_name}"
            try:
                raw = load_raw(blur_path, args.width, args.height, "Output")
            except Exception as exc:
                eprint(str(exc))
                return 1

        img = Image.frombytes("RGB", (args.width, args.height), raw)
        out_path = outdir / f"rgb_{loops}.png"
        img.save(out_path)
        images.append(img)
        print(f"Wrote: {out_path}")

    composite = Image.new("RGB", (args.width * 2, args.height))
    composite.paste(images[0], (0, 0))
    composite.paste(images[1], (args.width, 0))
    out_comp = outdir / f"rgb_{args.loops[0]}_{args.loops[1]}.png"
    composite.save(out_comp)
    print(f"Wrote: {out_comp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
