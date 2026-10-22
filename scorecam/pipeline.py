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

    def clock(self) -> str:
        """Rough position description, e.g. '4 o'clock', for comparing with notes."""
        if np.hypot(self.x_mm, self.y_mm) < 1.0:
            return "centre"
        ang = (np.degrees(np.arctan2(self.x_mm, -self.y_mm)) + 360) % 360
        h = int(round(ang / 30)) % 12
        return f"{12 if h == 0 else h} o'clock"


@dataclass
class CardResult:
    image: Path
    target: Target
    calibration: Calibration
    shots: list[Shot]
    rectified: np.ndarray
    T_mm_to_rect: np.ndarray
    debug: DetectionDebug

    @property
    def flags(self) -> list[str]:
        return list(self.calibration.flags)


def process_image(path: str | Path, target: Target, pellet_mm: float = 4.5) -> CardResult:
    path = Path(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return process_frame(bgr, target, pellet_mm, path)


def process_frame(bgr: np.ndarray, target: Target, pellet_mm: float = 4.5,
                  path: Path | None = None) -> CardResult:
    """The still pipeline on an in-memory BGR frame (live mode feeds frames here)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cal = calibrate(gray, target)
    rect, T = rectify(bgr, cal, target, PX_PER_MM)
    T_inv = np.linalg.inv(T)
    # two passes: the holes themselves disturb the ring measurement that the
    # detector relies on, so measure the rings again without them and detect again
    for _ in range(2):
        holes, dbg = detect_holes(rect, PX_PER_MM, T, target, cal, pellet_mm)
        positions = [T_inv @ np.array([h.x_mm, h.y_mm, 1.0]) for h in holes]
        update_ring_offsets(cal, gray, target, [(p[0], p[1], h.diameter_mm / 2) for p, h in zip(positions, holes)])
    shots = []
    for h, (x, y, _) in zip(holes, positions):
        r = float(np.hypot(x, y))
        # score against where the printed rings are near this hole (cancels paper deformation)
        offset = float(cal.ring_offset_at(r, np.arctan2(y, x))[0])
        score = score_hole(r - offset, target, pellet_mm)
        if score.value == 0 and r - offset - pellet_mm / 2 > target.outer_radius_mm:
            h.flags.append("off_target")  # outside the scoring rings: a miss, or more often not a hole
        shots.append(Shot(0, float(x), float(y), h, score, offset))
    # number shots by score, highest first (no shot order in a still image)
    shots.sort(key=lambda s: -s.score.decimal)
    for i, s in enumerate(shots, 1):
        s.index = i
    return CardResult(path, target, cal, shots, rect, T, dbg)
