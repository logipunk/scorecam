"""Still-image pipeline: image -> calibration -> rectify -> detect -> score."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calibrate import Calibration, calibrate, rectify, update_ring_offsets
from .detect import DetectionDebug, Hole, detect_holes
from .scoring import Score, score_hole
from .target import Target

PX_PER_MM = 10.0  # resolution of the rectified working image


@dataclass
class Shot:
    index: int
    x_mm: float  # target frame: origin at ring centre, x right, y down
    y_mm: float
    hole: Hole
    score: Score
    ring_offset_mm: float = 0.0  # local printed-ring displacement subtracted before scoring

    @property
    def flags(self) -> list[str]:
        return self.hole.flags

    def xy_corrected(self) -> tuple[float, float]:
        """Position with the local ring offset removed: where the hole sits relative
        to the printed rings, which is what stays put when the paper bends."""
        r = float(np.hypot(self.x_mm, self.y_mm))
        if r < 1e-6:
            return self.x_mm, self.y_mm
        k = (r - self.ring_offset_mm) / r
        return self.x_mm * k, self.y_mm * k

    def clock(self) -> str:
        """Rough position description, e.g. '4 o'clock', for comparing with notes."""
        if np.hypot(self.x_mm, self.y_mm) < 1.0:
            return "centre"
        ang = (np.degrees(np.arctan2(self.x_mm, -self.y_mm)) + 360) % 360
        h = int(round(ang / 30)) % 12
        return f"{12 if h == 0 else h} o'clock"


@dataclass
class FrameContext:
    """What a full read of a target leaves behind for fast reads of windows of it."""
    calibration: Calibration
    T: np.ndarray  # mm -> px of the full rectified image
    rect_shape: tuple
    card_mask: np.ndarray  # full rectified image, ignore regions applied
    card_theta: float
    paper_level: float

    def bounds_mm(self) -> tuple[float, float, float, float]:
        ppm = float(self.T[0, 0])
        return -self.T[0, 2] / ppm, -self.T[1, 2] / ppm, self.rect_shape[1] / ppm, self.rect_shape[0] / ppm


@dataclass
class CardResult:
    image: Path
    target: Target
    calibration: Calibration
    shots: list[Shot]
    rectified: np.ndarray
    T_mm_to_rect: np.ndarray
    debug: DetectionDebug
    context: FrameContext | None = None  # set by a full read; a window read carries its parent's
    window_mm: tuple | None = None  # (x0, y0, x1, y1) of a window read

    @property
    def flags(self) -> list[str]:
        return list(self.calibration.flags)


def process_image(path: str | Path, target: Target, pellet_mm: float = 4.5) -> CardResult:
    path = Path(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return process_frame(bgr, target, pellet_mm, path)


ROI_MARGIN_MM = 15.0  # live mode works on the rings plus this margin, not the whole card


def process_frame(bgr: np.ndarray, target: Target, pellet_mm: float = 4.5,
                  path: Path | None = None, roi: bool = False) -> CardResult:
    """The still pipeline on an in-memory BGR frame (live mode feeds frames here).
    roi=True restricts the work to the scoring rings plus ROI_MARGIN_MM."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cal = calibrate(gray, target)
    rect, T = rectify(bgr, cal, target, PX_PER_MM, roi_mm=ROI_MARGIN_MM if roi else None)
    T_inv = np.linalg.inv(T)
    # two passes: the holes themselves disturb the ring measurement that the
    # detector relies on, so measure the rings again without them and detect again
    for _ in range(2):
        holes, dbg = detect_holes(rect, PX_PER_MM, T, target, cal, pellet_mm)
        positions = [T_inv @ np.array([h.x_mm, h.y_mm, 1.0]) for h in holes]
        update_ring_offsets(cal, gray, target, [(p[0], p[1], h.diameter_mm / 2) for p, h in zip(positions, holes)])
    shots = _score(holes, positions, cal, target, pellet_mm)
    ctx = FrameContext(cal, T, rect.shape, dbg.card_mask, dbg.card_theta, dbg.paper_level)
    return CardResult(path, target, cal, shots, rect, T, dbg, ctx)


WINDOW_MARGIN_MM = 12.0  # a window read reaches this far beyond the changed area


def process_window(bgr: np.ndarray, target: Target, pellet_mm: float, ctx: FrameContext,
                   change_mm: tuple[float, float, float, float]) -> CardResult:
    """Fast read of one window of a target already read in full: the calibration,
    card outline and paper level of that read are reused, only the window is
    rectified and searched. Returns the holes inside the window."""
    fx0, fy0, fw, fh = ctx.bounds_mm()
    ppm = float(ctx.T[0, 0])
    x0 = max(fx0, change_mm[0] - WINDOW_MARGIN_MM)
    y0 = max(fy0, change_mm[1] - WINDOW_MARGIN_MM)
    x1 = min(fx0 + fw, change_mm[2] + WINDOW_MARGIN_MM)
    y1 = min(fy0 + fh, change_mm[3] + WINDOW_MARGIN_MM)
    # snap to the full image's pixel grid so the card mask can be cropped exactly
    px0, py0 = int((x0 - fx0) * ppm), int((y0 - fy0) * ppm)
    px1, py1 = int(np.ceil((x1 - fx0) * ppm)), int(np.ceil((y1 - fy0) * ppm))
    px1, py1 = min(px1, ctx.rect_shape[1]), min(py1, ctx.rect_shape[0])
    bounds = (fx0 + px0 / ppm, fy0 + py0 / ppm, (px1 - px0) / ppm, (py1 - py0) / ppm)
    rect, T = rectify(bgr, ctx.calibration, target, ppm, bounds_mm=bounds)
    card = ctx.card_mask[py0:py0 + rect.shape[0], px0:px0 + rect.shape[1]]
    holes, dbg = detect_holes(rect, ppm, T, target, ctx.calibration, pellet_mm, (card, ctx.card_theta), ctx.paper_level)
    T_inv = np.linalg.inv(T)
    positions = [T_inv @ np.array([h.x_mm, h.y_mm, 1.0]) for h in holes]
    shots = _score(holes, positions, ctx.calibration, target, pellet_mm)
    return CardResult(None, target, ctx.calibration, shots, rect, T, dbg, ctx, (x0, y0, x1, y1))


def _score(holes, positions, cal: Calibration, target: Target, pellet_mm: float) -> list[Shot]:
    shots = []
    for h, (x, y, _) in zip(holes, positions):
        r = float(np.hypot(x, y))
        # score against where the printed rings are near this hole (cancels paper deformation)
        offset = float(cal.ring_offset_at(r, np.arctan2(y, x))[0])
        score = score_hole(r - offset, target, pellet_mm)
        if score.value == 0 and r - offset - pellet_mm / 2 > target.outer_radius_mm:
            # outside the scoring rings: on the paper it is a miss (a shot scoring 0),
            # beyond the card edge it is not a hole at all
            cx, cy = target.center_offset_mm
            on_card = abs(x + cx) <= target.card_mm[0] / 2 and abs(y + cy) <= target.card_mm[1] / 2
            h.flags.append("miss" if on_card else "off_target")
        shots.append(Shot(0, float(x), float(y), h, score, offset))
    # number shots by score, highest first (no shot order in a still image)
    shots.sort(key=lambda s: -s.score.decimal)
    for i, s in enumerate(shots, 1):
        s.index = i
    return shots
