#!/usr/bin/env python3
"""Score one card image. Writes an annotated image to out/ and prints the shots.

Usage: scripts/score_card.py IMAGE [--target NAME] [--pellet MM] [--out DIR] [--cues]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scorecam.annotate import annotate, cue_sheet  # noqa: E402
from scorecam.pipeline import process_image  # noqa: E402
from scorecam.target import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--target", default="gamo_bone_collector_14cm")
    ap.add_argument("--pellet", type=float, default=4.5)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cues", action="store_true", help="also write the detection cue maps")
    a = ap.parse_args()
    target = load_target(a.target)
    res = process_image(a.image, target, a.pellet)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.image).stem
    cv2.imwrite(str(out / f"{stem}_annotated.jpg"), annotate(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if a.cues:
        cv2.imwrite(str(out / f"{stem}_cues.png"), cue_sheet(res))
    c = res.calibration
    print(f"{stem}: {c.mm_per_px:.4f} mm/px, ring fit RMS {c.residual_mm:.3f} mm, {c.n_inliers}/{c.n_points} pts, "
          f"max ring offset {np.abs(c.ring_offsets_smooth).max():.2f} mm" + (f"  FLAGS: {c.flags}" if c.flags else ""))
    for s in res.shots:
        print(f"  #{s.index} score {s.score.value:2d} ({s.score.decimal:4.1f})  d={s.score.distance_mm:5.2f} mm "
              f"at ({s.x_mm:+6.2f},{s.y_mm:+6.2f}) {s.clock():10s} hole {s.hole.diameter_mm:.1f} mm "
              f"round {s.hole.roundness:.2f} warm {s.hole.warm_frac:.2f} off {s.ring_offset_mm:+.2f} {' '.join(s.flags)}")
    for r in res.debug.rejected:
        print(f"  rejected: {r['why']} at ({r['x']:.0f},{r['y']:.0f}) d={r['diameter']:.1f} warm {r.get('warm_frac', 0):.2f}")


if __name__ == "__main__":
    main()
