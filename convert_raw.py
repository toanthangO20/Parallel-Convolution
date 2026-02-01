import argparse
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert RAW 8-bit grey/RGB file to PGM/PPM for viewing."
    )
    p.add_argument("input", help="Input .raw file")
    p.add_argument("width", type=int, help="Image width")
    p.add_argument("height", type=int, help="Image height")
    p.add_argument("mode", choices=["grey", "rgb"], help="Pixel format")
    p.add_argument(
        "-o",
        "--output",
        help="Output file (.pgm for grey, .ppm for rgb). Defaults to input name with .pgm/.ppm",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_path = args.input
    w = args.width
    h = args.height
    mode = args.mode

    if mode == "grey":
        expected = w * h
        magic = b"P5"
        ext = ".pgm"
    else:
        expected = w * h * 3
        magic = b"P6"
        ext = ".ppm"

    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    with open(in_path, "rb") as f:
        data = f.read()

    if len(data) != expected:
        print(
            f"Size mismatch: got {len(data)} bytes, expected {expected} (w={w}, h={h}, mode={mode})",
            file=sys.stderr,
        )
        return 2

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + ext

    header = magic + b"\n" + f"{w} {h}\n255\n".encode("ascii")

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(data)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
