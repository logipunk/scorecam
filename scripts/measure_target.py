#!/usr/bin/env python3
"""Measure printed ring radii on calibrated card images.

Calibrates each image against the given target definition (using all rings), then
reports where each printed circle actually sits in the mm frame. Use the median
offsets to correct a target YAML whose radii were estimated. Scale is set by the
target's own radii, so this refines relative radii only; absolute scale needs
calipers or a known card width.

Usage: scripts/measure_target.py TARGET IMAGE [IMAGE ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scorecam.calibrate import _locate, _sample_profiles, calibrate  # noqa: E402
from scorecam.target import load_target  # noqa: E402


def measure(gray, cal, target, window_mm=1.5, n_angles=360):
    found = {}
    for b in target.boundaries():
        prof, offs, model, px0, dirs, ppm = _sample_profiles(gray, cal.H_inv, b.radius_mm, window_mm, n_angles)
        offsets = []
        for i in range(n_angles):
            k = _locate(prof[i], b.kind)
            if k is not None:
                offsets.append(np.interp(k, np.arange(len(offs)), offs))
        found[b.radius_mm] = (float(np.median(offsets)) if offsets else float("nan"), len(offsets))
    return found


def main():
    target = load_target(sys.argv[1])
    per_ring = {b.radius_mm: [] for b in target.boundaries()}
    for path in sys.argv[2:]:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        cal = calibrate(gray, target)
        found = measure(gray, cal, target)
        print(f"{Path(path).name}: " + " ".join(f"{r:.1f}:{d:+.2f}({n})" for r, (d, n) in found.items()))
        for r, (d, n) in found.items():
            if n > 60:
                per_ring[r].append(d)
    print("\nring  model_mm  median_offset_mm  suggested_mm")
    for ring in target.rings:
        d = per_ring[ring.radius_mm]
        med = float(np.median(d)) if d else float("nan")
        print(f"{ring.value:4d}  {ring.radius_mm:8.2f}  {med:+16.3f}  {ring.radius_mm + med:12.2f}")


if __name__ == "__main__":
    main()
