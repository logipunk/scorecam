#!/usr/bin/env python3
"""Replay recorded frames through the live scorer and print the events.

Usage: scripts/live_replay.py --target NAME [--caliber MM] [--session DIR] FRAME...
Frames are fed in the order given (sort them yourself, e.g. a glob). With
--session, a session directory is written exactly as the Pi service would.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorecam.live import LiveScorer  # noqa: E402
from scorecam.session import SessionHeader, SessionWriter, ShotRecord  # noqa: E402
from scorecam.target import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--target", required=True)
    ap.add_argument("--caliber", type=float, default=4.5)
    ap.add_argument("--session", help="write a session directory here")
    ap.add_argument("--stable", type=int, default=3, help="still frames before analysing")
    ap.add_argument("--repeat", type=int, default=1, help="feed each file this many times (a still image is a still scene)")
    a = ap.parse_args()
    target = load_target(a.target)
    scorer = LiveScorer(target, a.caliber, stable_frames=a.stable)
    writer = None
    if a.session:
        header = SessionHeader(id=time.strftime("%Y%m%d_%H%M%S") + "_replay", target=target.name,
                               caliber_mm=a.caliber, started=time.time())
        writer = SessionWriter(Path(a.session), header)
    total = 0
    for i, f in enumerate(f for f in a.frames for _ in range(a.repeat)):
        bgr = cv2.imread(f)
        if bgr is None:
            print(f"[{i}] {f}: unreadable", file=sys.stderr)
            continue
        t0 = time.time()
        ev = scorer.feed(bgr, t=float(i))
        if ev is None:
            continue
        took = time.time() - t0
        line = f"[{i:3d}] {Path(f).name:28s} {ev.kind:7s} series {ev.series}"
        if ev.kind == "locked":
            line += f"  existing holes {ev.existing}  (cal {ev.result.calibration.mm_per_px:.3f} mm/px, {took:.1f}s)"
        elif ev.kind == "shots":
            total += len(ev.shots)
            line += "  NEW: " + ", ".join(f"{s.score.value}{'X' if s.score.inner else ''} ({s.score.decimal:.1f}) {s.clock()}"
                                          + (f" [{','.join(s.flags)}]" if s.flags else "") for s in ev.shots)
            line += f"  ({took:.1f}s)"
        elif ev.message:
            line += f"  {ev.message}"
        print(line)
        if writer and ev.kind in ("locked", "shots", "lost"):
            frame = writer.save_frame(bgr, ev.t) if ev.result is not None else ""
            if ev.kind == "shots":
                writer.shots([ShotRecord(0, ev.series, ev.t, s.x_mm, s.y_mm, s.score.value, s.score.decimal,
                                         s.score.inner, list(s.flags), frame) for s in ev.shots], ev.t)
            else:
                writer.event(ev.kind, ev.t, series=ev.series, existing=ev.existing, frame=frame, message=ev.message)
    print(f"total new shots: {total}")
    if writer:
        writer.close()
        print(f"session written to {writer.dir}")


if __name__ == "__main__":
    main()
