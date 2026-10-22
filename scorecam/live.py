"""Live mode: a stream of frames in, shot events out.

The camera watches the target. Three things can happen to the picture:
  * it moves (target turning, someone walking past, exposure hunting),
  * it holds still with no rings in view (target edge-on, nothing mounted),
  * it holds still with the rings in view ("locked").

Only a still, locked picture is analysed, and only when it has just become
still. A new hole changes the picture, so it ends a still period and starts
another; a turning target does the same. Either way the first still frame
after the change is compared with the last analysed one, and the holes that
were not there before are the new shots. This covers turning bullseye targets
(each face-on period is one string) and fixed 10 m targets (each shot is its
own change) with the same logic.

Everything here is pure: feed frames, get events. No camera, no threads.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import CalibrationError
from .pipeline import CardResult, Shot, process_frame
from .target import Target

MOTION_WIDTH = 320  # frames are compared at this width (a 4.5 mm hole at 0.07 mm/px is ~5 px wide here)
MOTION_BLUR = 5  # box blur (px, at the reduced scale) before differencing: kills sensor noise
MOTION_LEVEL = 24  # gray levels a blurred pixel must change by to count as changed
MOTION_MIN_PIXELS = 12  # changed pixels (reduced scale) that make a frame "moving"
STABLE_FRAMES = 3  # consecutive still frames before a picture is analysed
MATCH_MM = 2.5  # a hole this close to a known one is the same hole
NEW_TARGET_GONE_FRAC = 0.5  # if more than this share of known holes vanish, a new target was hung


@dataclass
class LiveEvent:
    kind: str  # "locked" | "lost" | "shots" | "moving" | "still"
    t: float
    series: int = 0
    shots: list[Shot] = field(default_factory=list)  # new shots (kind == "shots")
    existing: int = 0  # holes already on the target when it locked (kind == "locked")
    result: CardResult | None = None  # the analysed frame (locked / shots)
    message: str = ""


def frame_motion(prev_small: np.ndarray, cur_small: np.ndarray) -> int:
    """Number of changed pixels between two reduced grayscale frames."""
    d = cv2.absdiff(prev_small, cur_small)
    d = cv2.blur(d, (MOTION_BLUR, MOTION_BLUR))
    return int(np.count_nonzero(d > MOTION_LEVEL))


def reduce(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    f = MOTION_WIDTH / gray.shape[1]
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray


def new_holes(reference: list[Shot], current: list[Shot], tol_mm: float = MATCH_MM) -> list[Shot]:
    """Shots in `current` with no counterpart in `reference` (greedy nearest match)."""
    pool = list(reference)
    fresh = []
    for s in sorted(current, key=lambda s: -s.score.decimal):
        if not pool:
            fresh.append(s)
            continue
        d = [np.hypot(s.x_mm - r.x_mm, s.y_mm - r.y_mm) for r in pool]
        i = int(np.argmin(d))
        if d[i] <= tol_mm:
            pool.pop(i)
        else:
            fresh.append(s)
    return fresh


class LiveScorer:
    """Feed frames in order; returns an event when something happened."""

    def __init__(self, target: Target, pellet_mm: float = 4.5, stable_frames: int = STABLE_FRAMES):
        self.target = target
        self.pellet_mm = pellet_mm
        self.stable_frames = stable_frames
        self.prev_small: np.ndarray | None = None
        self.still_count = 0
        self.analysed_still = False  # the current still period has been analysed
        self.analysis_due = False  # set by observe(): a full frame should be analysed now
        self.locked = False
        self.reference: list[Shot] | None = None  # holes known to be on the target
        self.series = 0
        self.last_result: CardResult | None = None

    def feed(self, bgr: np.ndarray, t: float | None = None, full=None) -> LiveEvent | None:
        """observe() then analyse() in one call. `bgr` is watched for motion (any
        resolution, e.g. the camera's low-res stream). When a still picture is to
        be analysed, `full` supplies the full-resolution frame: an array, a
        callable returning one, or None to use `bgr` itself."""
        t = time.time() if t is None else t
        ev = self.observe(bgr, t)
        if ev is not None or not self.analysis_due:
            return ev
        frame = full() if callable(full) else (bgr if full is None else full)
        return self.analyse(frame, t)

    def observe(self, bgr: np.ndarray, t: float | None = None) -> LiveEvent | None:
        """Track motion. Afterwards `analysis_due` says whether a full-resolution
        frame of this still picture should now go through analyse()."""
        t = time.time() if t is None else t
        self.analysis_due = False
        small = reduce(bgr)
        moving = self.prev_small is not None and frame_motion(self.prev_small, small) >= MOTION_MIN_PIXELS
        self.prev_small = small
        if moving:
            was_still = self.still_count >= self.stable_frames
            self.still_count = 0
            self.analysed_still = False
            return LiveEvent("moving", t, series=self.series) if was_still else None
        self.still_count += 1
        if self.still_count >= self.stable_frames and not self.analysed_still:
            self.analysis_due = True
        return None

    def analyse(self, frame: np.ndarray, t: float | None = None) -> LiveEvent | None:
        """Run the still pipeline on a full frame of the current still picture."""
        t = time.time() if t is None else t
        self.analysed_still = True
        self.analysis_due = False
        return self._analyse(frame, t)

    def _analyse(self, bgr: np.ndarray, t: float) -> LiveEvent | None:
        try:
            result = process_frame(bgr, self.target, self.pellet_mm, roi=True)
        except CalibrationError as e:
            if self.locked:
                self.locked = False
                return LiveEvent("lost", t, series=self.series, message=str(e))
            return LiveEvent("still", t, message=f"no target: {e}")
        self.last_result = result
        current = [s for s in result.shots if "off_target" not in s.flags]
        if not self.locked:
            self.locked = True
            self.series += 1
            if self.reference is None:  # first lock of the session: what is on the paper now is not ours
                self.reference = current
                return LiveEvent("locked", t, series=self.series, existing=len(current), result=result)
        fresh = new_holes(self.reference, current)
        gone = new_holes(current, self.reference)  # known holes no longer seen
        if len(self.reference) >= 2 and len(gone) > len(self.reference) * NEW_TARGET_GONE_FRAC:
            # most known holes have vanished: a fresh target was hung, not a string fired
            self.reference = current
            return LiveEvent("locked", t, series=self.series, existing=len(current), result=result,
                             message="new target")
        self.reference = current if fresh else self.reference
        if fresh:
            return LiveEvent("shots", t, series=self.series, shots=fresh, result=result)
        return LiveEvent("still", t, series=self.series, result=result)
