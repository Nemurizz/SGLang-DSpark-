"""Independent conservative W8A8 FP8 route, same immutable quality protocol."""
import argparse
from pathlib import Path
from quality_cudnn_suite import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    args=p.parse_args()
    raise SystemExit(run(args.root, args.previous, 'w8a8_fp8_native'))
