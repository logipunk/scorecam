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
MATCH_MM = 3.0  # a hole this close to a known one is the same hole
ROTATION_REACH_MM = 8.0  # pairs closer than this vote on the sheet rotation between frames
NEW_TARGET_GONE_FRAC = 0.5  # if more than this share of known holes vanish, a new target was hung
LOCK_ANALYSES = 2  # still frames analysed before the first lock; their holes are united


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


def _xy(shots: list[Shot]) -> np.ndarray:
    return np.array([s.xy_corrected() for s in shots], float).reshape(-1, 2)


def align_rotation(reference: list[Shot], current: list[Shot], reach_mm: float = ROTATION_REACH_MM) -> float:
    """Rotation (radians) that best maps current onto reference, from the holes
    themselves. Rings give no angular reference, so a sheet re-hung at a slightly
    different angle rotates every hole about the centre; the holes it shares with
    the reference reveal the angle (weighted median of pairwise angle differences)."""
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    R, C = _xy(reference), _xy(current)
    ang, w = [], []
    for c in C:
        d = np.hypot(R[:, 0] - c[0], R[:, 1] - c[1])
        i = int(np.argmin(d))
        rc, rr = np.hypot(*c), np.hypot(*R[i])
        if d[i] < reach_mm and min(rc, rr) > 5.0:
            delta = (np.arctan2(R[i, 1], R[i, 0]) - np.arctan2(c[1], c[0]) + np.pi) % (2 * np.pi) - np.pi
            ang.append(delta)
            w.append(min(rc, rr))  # outer holes pin the angle down better
    if len(ang) < 2:
        return 0.0
    order = np.argsort(ang)
    cum = np.cumsum(np.array(w)[order])
    return float(np.array(ang)[order][np.searchsorted(cum, cum[-1] / 2)])


def new_holes(reference: list[Shot], current: list[Shot], tol_mm: float = MATCH_MM) -> list[Shot]:
    """Shots in `current` with no counterpart in `reference` (greedy nearest match
    on ring-corrected positions, after removing any rotation between the two).
    An overlap group (one blob split into n estimated holes) is matched as a
    group: it yields new shots only for the members beyond the known holes it
    covers, so a torn hole that splits differently from frame to frame does not
    come back as a new shot."""
    if not reference:
        return list(current)
    rot = align_rotation(reference, current)
    c, s_ = np.cos(rot), np.sin(rot)
    ref_xy = _xy(reference)

    def pos(shot):
        x, y = shot.xy_corrected()
        return np.array([c * x - s_ * y, s_ * x + c * y])

    taken = np.zeros(len(reference), bool)
    fresh = []
    singles = [s for s in current if s.hole is None or s.hole.group < 0]
    groups: dict = {}
    for s in current:
        if s.hole is not None and s.hole.group >= 0:
            groups.setdefault(s.hole.group, []).append(s)
    for shot in sorted(singles, key=lambda s: -s.score.decimal):
        p = pos(shot)
        d = np.hypot(ref_xy[:, 0] - p[0], ref_xy[:, 1] - p[1])
        d[taken] = np.inf
        i = int(np.argmin(d)) if len(d) else -1
        if i >= 0 and d[i] <= tol_mm:
            taken[i] = True
        else:
            fresh.append(shot)
    for members in groups.values():
        P = np.array([pos(m) for m in members])
        covered = []
        for i in np.flatnonzero(~taken):  # known holes this group lies over
            if np.hypot(P[:, 0] - ref_xy[i, 0], P[:, 1] - ref_xy[i, 1]).min() <= tol_mm:
                covered.append(i)
        for i in covered:
            taken[i] = True
        extra = len(members) - len(covered)
        if extra > 0:
            # the members farthest from any known hole are the new ones
            far = [np.hypot(ref_xy[covered, 0] - p[0], ref_xy[covered, 1] - p[1]).min() if covered else np.inf for p in P]
            order = np.argsort(far)[::-1]
            fresh.extend(members[k] for k in order[:extra])
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
        self.pending_lock: list[Shot] | None = None  # holes of the first lock analysis, awaiting the second
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
            if self.reference is None and self.pending_lock is None and LOCK_ANALYSES > 1:
                # first lock of the session: look twice before deciding what is already on the paper
                self.pending_lock = current
                self.analysed_still = False  # analyse the next still frame as well
                return None
            self.locked = True
            self.series += 1
            if self.reference is None:  # first lock of the session: what is on the paper now is not ours
                known = list(self.pending_lock or [])
                self.reference = known + new_holes(known, current)
                self.pending_lock = None
                return LiveEvent("locked", t, series=self.series, existing=len(self.reference), result=result)
        fresh = new_holes(self.reference, current)
        gone = new_holes(current, self.reference)  # known holes no longer seen
        if len(self.reference) >= 2 and len(gone) > len(self.reference) * NEW_TARGET_GONE_FRAC:
            # most known holes have vanished: a fresh target was hung, not a string fired
            self.reference = current
            return LiveEvent("locked", t, series=self.series, existing=len(current), result=result,
                             message="new target")
        if fresh:  # known holes are kept even when a frame misses one: a flickering detection must not come back as new
            self.reference = list(self.reference) + fresh
            return LiveEvent("shots", t, series=self.series, shots=fresh, result=result)
        return LiveEvent("still", t, series=self.series, result=result)
