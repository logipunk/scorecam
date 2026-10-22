#!/usr/bin/env python3
"""Run every card in test-cards/ground_truth.yaml and compare with the hand scores.

Ground truth gives, per card, a list of holes with a free-text `seen` hint and a
`score` (integer ring, 0 = flier on the card, null = not yet scored by hand).
Matching: expected holes with a score are matched to found holes with the same
score; the rest are matched by count only. Writes annotated images to out/.

Usage: scripts/evaluate.py [--target NAME] [--pellet MM] [--out DIR] [--cues] [--gt FILE]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scorecam.annotate import annotate, cue_sheet  # noqa: E402
from scorecam.calibrate import CalibrationError  # noqa: E402
from scorecam.pipeline import process_image  # noqa: E402
from scorecam.target import load_target  # noqa: E402

SKIP = {"test.jpg", "hole_crop.jpg"}


def match(expected: list, found: list) -> tuple[list, list, list]:
    """Returns (pairs [(exp, shot)], misses [exp], false_positives [shot])."""
    pairs, pool = [], list(found)
    rest = []
    for e in expected:  # scored holes first: match on score
        if e.get("score") is None:
            rest.append(e)
            continue
        want_x = e.get("inner")
        hit = next((s for s in pool if s.score.value == e["score"] and (want_x is None or bool(s.score.inner) == want_x)), None)
        if hit is None:
            rest.append(e)
        else:
            pairs.append((e, hit))
            pool.remove(hit)
    misses = []
    for e in rest:  # unscored: any remaining on-target detection counts
        on_target = [s for s in pool if "off_target" not in s.flags]
        if on_target:
            pairs.append((e, on_target[0]))
            pool.remove(on_target[0])
        else:
            misses.append(e)
    return pairs, misses, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="target name; default: from ground truth")
    ap.add_argument("--pellet", "--caliber", dest="pellet", type=float, default=None,
                    help="projectile diameter in mm; default from the ground truth file")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cues", action="store_true")
    ap.add_argument("--gt", default="test-cards/ground_truth.yaml")
    a = ap.parse_args()
    gt_path = Path(a.gt)
    gt = yaml.safe_load(open(gt_path))
    target = load_target(a.target or gt["target"]["name"])
    pellet = a.pellet or float(gt["target"].get("pellet_mm", 4.5))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    totals = dict(expected=0, found=0, matched=0, score_checked=0, score_ok=0, missed=0, fp=0, off_target=0, flagged=0)
    for card in gt["cards"]:
        if card["file"] in SKIP:
            continue
        path = gt_path.parent / card["file"]
        expected = card.get("holes", [])
        t0 = time.time()
        try:
            res = process_image(path, target, pellet)
        except CalibrationError as e:
            print(f"\n{card['file']}: CALIBRATION FAILED: {e}")
            totals["expected"] += len(expected)
            totals["missed"] += len(expected)
            continue
        dt = time.time() - t0
        cv2.imwrite(str(out / f"{path.stem}_annotated.jpg"), annotate(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if a.cues:
            cv2.imwrite(str(out / f"{path.stem}_cues.png"), cue_sheet(res))
        pairs, misses, fps = match(expected, res.shots)
        checked = [(e, s) for e, s in pairs if e.get("score") is not None]
        ok = [(e, s) for e, s in checked if e["score"] == s.score.value and (e.get("inner") is None or bool(s.score.inner) == e["inner"])]
        c = res.calibration
        print(f"\n{card['file']}: {len(expected)} expected, {len(res.shots)} found, {dt:.1f}s  "
              f"[{c.mm_per_px:.4f} mm/px, ring RMS {c.residual_mm:.2f} mm"
              + (f", FLAGS {','.join(c.flags)}" if c.flags else "") + "]")
        if card.get("notes"):
            print(f"  notes: {card['notes']}")
        for e, s in pairs:
            verdict = ("OK" if e["score"] == s.score.value else "WRONG") if e.get("score") is not None else "unverified"
            print(f"  {verdict:10s} expected {str(e.get('score')):4s} got {s.score.value:2d} ({s.score.decimal:4.1f}) "
                  f"{s.clock():10s} d={s.score.distance_mm:5.1f} mm  seen: {e.get('seen', '')!r}"
                  + (f"  [{', '.join(s.flags)}]" if s.flags else ""))
        for e in misses:
            print(f"  MISSED     expected {str(e.get('score')):4s}  seen: {e.get('seen', '')!r}")
        for s in fps:
            print(f"  {'OFF-TARGET' if 'off_target' in s.flags else 'FALSE POS '} got {s.score.value:2d} ({s.score.decimal:4.1f}) {s.clock():10s} d={s.score.distance_mm:5.1f} mm"
                  + (f"  [{', '.join(s.flags)}]" if s.flags else ""))
        totals["expected"] += len(expected)
        totals["found"] += len(res.shots)
        totals["matched"] += len(pairs)
        totals["score_checked"] += len(checked)
        totals["score_ok"] += len(ok)
        totals["missed"] += len(misses)
        totals["fp"] += sum(1 for s in fps if "off_target" not in s.flags)
        totals["off_target"] += sum(1 for s in fps if "off_target" in s.flags)
        totals["flagged"] += sum(1 for s in res.shots if s.flags)
    t = totals
    print("\nSUMMARY")
    print(f"  holes expected {t['expected']}, found {t['found']}, matched {t['matched']}, "
          f"missed {t['missed']}, false positives {t['fp']}, off-target detections {t['off_target']}, "
          f"flagged for review {t['flagged']}")
    print(f"  hand-scored holes: {t['score_ok']}/{t['score_checked']} correct"
          + ("" if t["score_checked"] else " (no integer scores filled in yet)"))
    print(f"  annotated images in {out}/")


if __name__ == "__main__":
    main()
