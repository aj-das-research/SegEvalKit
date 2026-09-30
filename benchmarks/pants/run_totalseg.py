"""Run the official TotalSegmentator (task "total") on a shard of the PanTS test set."""
import argparse
import os
import sys
import time
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--parts", type=int, default=1)
ap.add_argument("--part", type=int, default=0)
a = ap.parse_args()
cases = sorted(os.listdir(a.images))[a.part::a.parts]
for i, c in enumerate(cases):
    out = Path(a.out) / c
    if (out / "liver.nii.gz").exists():
        continue
    t = time.time()
    totalsegmentator(str(Path(a.images) / c / "ct.nii.gz"), str(out), task="total", quiet=True, device="gpu")
    print(f"[{i + 1}/{len(cases)}] {c} {time.time() - t:.1f}s", flush=True)
