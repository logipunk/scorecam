"""Debug drawings: detected rings, centre and scored holes on the rectified card."""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import PX_PER_MM, CardResult

GREEN, RED, YELLOW, CYAN, ORANGE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 200, 0), (0, 140, 255)


def annotate(result: CardResult) -> np.ndarray:
    img = result.rectified.copy()
    T = result.T_mm_to_rect
    s = PX_PER_MM

    def pt(x_mm, y_mm):
        p = T @ np.array([x_mm, y_mm, 1.0])
        return int(round(p[0])), int(round(p[1]))

    c = pt(0, 0)
    cal = result.calibration
    th = np.linspace(0, 2 * np.pi, 181)
    for ring in result.target.rings:
        cv2.circle(img, c, int(ring.radius_mm * s), (160, 160, 160), 1)  # model
        if cal.ring_offsets_smooth is not None:  # measured printed ring
            rr = ring.radius_mm + cal.ring_offset_at(np.full_like(th, ring.radius_mm), th)
            pts = np.stack([T[0, 2] + rr * np.cos(th) * s, T[1, 2] + rr * np.sin(th) * s], axis=1)
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, GREEN, 1)
    cv2.drawMarker(img, c, GREEN, cv2.MARKER_CROSS, 30, 1)
    for shot in result.shots:
        p = pt(shot.x_mm, shot.y_mm)
        color = ORANGE if shot.flags else (YELLOW if shot.score.value else CYAN)
        cv2.circle(img, p, int(shot.hole.diameter_mm / 2 * s), color, 2)
        cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, 12, 1)
        label = f"#{shot.index} {shot.score.decimal:.1f}" + (" " + ",".join(shot.flags) if shot.flags else "")
        cv2.putText(img, label, (p[0] + 28, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for r in result.debug.rejected:
        p = (int(r["x"]), int(r["y"]))
        cv2.circle(img, p, max(4, int(r["diameter"] / 2 * s)), RED, 1)
        cv2.putText(img, r["why"], (p[0] + 10, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
    txt = (f"{result.image.name}  {cal.mm_per_px:.4f} mm/px  ring fit RMS {cal.residual_mm:.3f} mm "
           f"({cal.n_inliers}/{cal.n_points} pts)" + (f"  FLAGS: {', '.join(cal.flags)}" if cal.flags else ""))
    cv2.putText(img, txt, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return img


def cue_sheet(result: CardResult) -> np.ndarray:
    """Side-by-side view of the detection cue maps."""
    d = result.debug
    maps = [d.dark, d.bright, d.warm, d.candidates, d.numeral_strokes]
    tiles = [cv2.cvtColor((m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR) for m in maps]
    for t, name in zip(tiles, ["dark", "bright", "warm", "candidates", "numeral strokes"]):
        cv2.putText(t, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return np.hstack(tiles)
