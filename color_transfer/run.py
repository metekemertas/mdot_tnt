"""Run color transfer between the two bundled asset images."""

import argparse
import time
from pathlib import Path

import torch as th

from color_transfer import transfer_colors

ASSETS = Path(__file__).parent / "assets"

_DEFAULT_DEVICE = "cuda" if th.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="OT color transfer (low-memory point-cloud mode)")
    parser.add_argument("--max-pixels", type=int, default=8192, help="Max pixels per image (default: 8192)")
    parser.add_argument("--block-size", type=int, default=512, help="Column block size for the low-memory solver (default: 512)")
    parser.add_argument("--gamma-f", type=float, default=1024.0, help="OT temperature / inverse regularization (default: 1024)")
    parser.add_argument("--color-space", choices=["lab", "rgb"], default="lab", help="Working color space (default: lab)")
    parser.add_argument("--device", default=_DEFAULT_DEVICE, help=f"Torch device (default: {_DEFAULT_DEVICE})")
    parser.add_argument("--both-directions", action="store_true", help="Also transfer in the reverse direction (2 → 1)")
    args = parser.parse_args()

    src = ASSETS / "1.webp"
    tgt = ASSETS / "2.webp"

    kwargs = dict(
        max_pixels=args.max_pixels,
        block_size=args.block_size,
        gamma_f=args.gamma_f,
        color_space=args.color_space,
        device=args.device,
    )

    print(f"max_pixels={args.max_pixels}  block_size={args.block_size}  "
          f"gamma_f={args.gamma_f}  color_space={args.color_space}  device={args.device}")

    print("\n1 → 2  (applying target palette of 2.webp onto 1.webp) …")
    t0 = time.perf_counter()
    out = transfer_colors(src, tgt, **kwargs)
    elapsed = time.perf_counter() - t0
    out_path = ASSETS / "out.png"
    out.save(out_path)
    print(f"  done in {elapsed:.1f}s  →  {out_path}")

    if args.both_directions:
        print("\n2 → 1  (applying target palette of 1.webp onto 2.webp) …")
        t0 = time.perf_counter()
        out_rev = transfer_colors(tgt, src, **kwargs)
        elapsed = time.perf_counter() - t0
        out_rev_path = ASSETS / "out_reverse.png"
        out_rev.save(out_rev_path)
        print(f"  done in {elapsed:.1f}s  →  {out_rev_path}")


if __name__ == "__main__":
    main()
