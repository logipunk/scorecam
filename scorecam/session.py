"""Session data model: what a shooting session records, independent of how the
shots were detected (scoring camera now, aim-trace sensor later).

A session is a directory:
    session.json   header: target, caliber, shooter, start time
    events.jsonl   one JSON object per event, in time order (locked, shots, lost, note)
    frames/        the analysed frames the shots were found in (JPEG)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ShotRecord:
    index: int  # 1-based within the session
    series: int  # 1-based; a series is one face-on period of the target (one string)
    t: float  # unix time the frame holding the shot was analysed
    x_mm: float
    y_mm: float
    value: int
    decimal: float
    inner: bool
    flags: list[str] = field(default_factory=list)
    frame: str = ""  # frames/<name>.jpg the shot was found in
    source: str = "camera"  # or "manual" (corrected / added by the user)


@dataclass
class SessionHeader:
    id: str
    target: str
    caliber_mm: float
    started: float
    shooter: str = ""
    notes: str = ""


class SessionWriter:
    """Appends events to a session directory. Cheap to call from the live loop."""

    def __init__(self, root: Path, header: SessionHeader):
        self.dir = Path(root) / header.id
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.header = header
        (self.dir / "session.json").write_text(json.dumps(asdict(header), indent=2))
        self._events = open(self.dir / "events.jsonl", "a")
        self.n_shots = 0

    def event(self, kind: str, t: float | None = None, **data) -> dict:
        ev = {"kind": kind, "t": t if t is not None else time.time(), **data}
        self._events.write(json.dumps(ev) + "\n")
        self._events.flush()
        return ev

    def save_frame(self, bgr: np.ndarray, t: float) -> str:
        name = f"frames/{t:.3f}.jpg"
        cv2.imwrite(str(self.dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return name

    def shots(self, records: list[ShotRecord], t: float) -> dict:
        for r in records:
            self.n_shots += 1
            r.index = self.n_shots
        return self.event("shots", t, shots=[asdict(r) for r in records])

    def close(self) -> None:
        self._events.close()


def read_session(path: Path) -> tuple[SessionHeader, list[dict]]:
    path = Path(path)
    header = SessionHeader(**json.loads((path / "session.json").read_text()))
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line.strip()]
    return header, events


def session_shots(events: list[dict]) -> list[ShotRecord]:
    out = []
    for ev in events:
        if ev["kind"] == "shots":
            out.extend(ShotRecord(**s) for s in ev["shots"])
    return out
