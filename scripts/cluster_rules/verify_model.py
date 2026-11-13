#!/usr/bin/env python3
"""Verify separately staged model files against the recorded L40 model identity."""
import argparse
import json
from pathlib import Path
from common import digest

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--model', type=Path, required=True)
    args = ap.parse_args()
    identity = json.loads((args.root / 'model_identity.json').read_text())
    for name, expected in identity['files'].items():
        if digest(args.model / name) != expected:
            raise ValueError(f'Model file hash differs: {name}')
        print(f'OK {name}', flush=True)
    print('Model identity verified.')
