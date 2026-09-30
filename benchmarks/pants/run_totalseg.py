"""Run the official TotalSegmentator (task "total") on a shard of the PanTS test set."""
import argparse
import os
import sys
import time
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator



def main():
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
        for attempt in range(3):  # worker pools can die if the node's /dev/shm is cleaned under us
            try:
                totalsegmentator(str(Path(a.images) / c / "ct.nii.gz"), str(out), task="total", quiet=True,
                                 device="gpu")
                break
            except (RuntimeError, FileNotFoundError, OSError) as exc:
                print(f"retry {attempt + 1} for {c}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(10)
        print(f"[{i + 1}/{len(cases)}] {c} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":  # TotalSegmentator's nnU-Net workers use spawn: the guard is required
    main()
