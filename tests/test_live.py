"""Live scorer on a synthetic frame stream: clean target locks, the target turns
away (no rings) and comes back with five holes, then again with five more; each
face-on period yields exactly its new shots. Then the same on a fixed target
where holes appear one at a time. Also: two real captures of the same sheet
produce no new shots."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from scorecam.live import LiveScorer, frame_motion, new_holes, reduce
from scorecam.pipeline import Shot
from scorecam.scoring import Score
from scorecam.target import load_target
from tests.test_pipeline_synthetic import photograph, render

STRING1 = [(2.0, 1.0), (-15.0, 4.0), (10.0, -12.0), (28.0, 16.0), (-50.0, -5.0)]  # none on the black edge (38.9 mm)
STRING2 = [(0.0, 25.0), (-22.0, -18.0), (48.0, 0.0), (5.0, -45.0), (-60.0, 30.0)]


def frame(target, holes, tmp_path, caliber=5.6):
    return cv2.imread(str(photograph(render(target, holes, caliber), tmp_path)))


def edge_on(like):
    """What the camera sees while the target is turned away: dark background only."""
    return np.full_like(like, (60, 50, 40))


def run(scorer, frames):
    return [ev for ev in (scorer.feed(f, t=float(i)) for i, f in enumerate(frames)) if ev is not None]


def test_turning_target_strings(tmp_path):
    target = load_target("nra_b3")
    clean = frame(target, [], tmp_path)
    five = frame(target, STRING1, tmp_path)
    ten = frame(target, STRING1 + STRING2, tmp_path)
    away = edge_on(clean)
    frames = [clean] * 4 + [away] * 4 + [five] * 4 + [away] * 4 + [ten] * 4
    events = run(LiveScorer(target, 5.6), frames)
    kinds = [e.kind for e in events]
    assert kinds == ["locked", "moving", "lost", "moving", "shots", "moving", "lost", "moving", "shots"], kinds
    assert events[0].existing == 0
    s1, s2 = events[4], events[8]
    assert s1.series == 2 and s2.series == 3
    assert len(s1.shots) == 5 and len(s2.shots) == 5
    for shots, truth in ((s1.shots, STRING1), (s2.shots, STRING2)):
        for (x, y) in truth:
            assert min(np.hypot(s.x_mm - x, s.y_mm - y) for s in shots) < 1.0


def test_fixed_target_one_shot_at_a_time(tmp_path):
    target = load_target("issf_10m_air_pistol")
    holes = [(1.0, -1.0), (0.0, 14.0), (-30.0, 0.0)]
    frames = [frame(target, [], tmp_path, 4.5)] * 4
    for n in range(1, len(holes) + 1):  # the first changed frame counts as motion, then 3 still ones
        frames += [frame(target, holes[:n], tmp_path, 4.5)] * 4
    events = run(LiveScorer(target, 4.5), frames)
    shots = [e for e in events if e.kind == "shots"]
    assert [len(e.shots) for e in shots] == [1, 1, 1]
    assert [e.shots[0].score.value for e in shots] == [10, 9, 7]
    assert all(e.series == 1 for e in shots)  # the target never turned


def test_still_frames_do_not_reanalyse(tmp_path):
    target = load_target("nra_b2")
    f = frame(target, [(0, 0)], tmp_path)
    events = run(LiveScorer(target, 5.6), [f] * 10)
    assert [e.kind for e in events] == ["locked"]


def test_motion_metric_ignores_noise():
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (760, 1014), dtype=np.uint8)
    noisy = np.clip(base.astype(int) + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    assert frame_motion(base, noisy) == 0
    hole = noisy.copy()
    cv2.circle(hole, (500, 380), 8, 0, -1)  # a 4.5 mm hole at ~0.07 mm/px, reduced 4x
    assert frame_motion(base, hole) >= 12


def test_new_holes_matching():
    def shot(x, y):
        return Shot(0, x, y, None, Score(10, 10.0, False, 0.0))
    ref = [shot(0, 0), shot(10, 10)]
    cur = [shot(0.3, -0.2), shot(10.4, 9.8), shot(-20, 5)]
    fresh = new_holes(ref, cur)
    assert [(s.x_mm, s.y_mm) for s in fresh] == [(-20, 5)]


REAL = Path("test-cards/nra")


@pytest.mark.skipif(not (REAL / "20261022_132336.jpg").exists(), reason="real captures not present")
def test_same_sheet_twice_gives_no_shots():
    target = load_target("nra_b2")
    a = cv2.imread(str(REAL / "20261022_132336.jpg"))
    b = cv2.imread(str(REAL / "20261022_132339.jpg"))
    events = run(LiveScorer(target, 5.6, stable_frames=1), [a, b])
    assert events[0].kind == "locked" and events[0].existing >= 7
    assert all(e.kind != "shots" for e in events[1:]), [e.kind for e in events]
