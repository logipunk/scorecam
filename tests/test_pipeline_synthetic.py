"""End-to-end check on rendered targets: draw a target from its definition, punch
known holes, photograph it with a perspective warp, and see that the pipeline
calibrates and scores them. Validates the target files as much as the code."""
import cv2
import numpy as np
import pytest

from scorecam.pipeline import process_image
from scorecam.target import Target, load_target

PX = 6  # render resolution, px per mm
PAPER, INK, TAN, DARK = 225, 35, (110, 150, 175), (20, 20, 20)  # tan rim ~0.5 of paper (real: 0.26-0.33 above ink)


def render(target: Target, holes_mm, caliber_mm):
    w, h = (int(d * PX) for d in target.card_mm)
    img = np.full((h, w, 3), PAPER, np.uint8)
    c = (w // 2, h // 2)
    for ring in reversed(target.rings):  # outer first
        color = (INK,) * 3 if ring.color == "black" else (PAPER,) * 3
        cv2.circle(img, c, int(ring.radius_mm * PX), color, -1)
        line = (PAPER,) * 3 if ring.color == "black" else (INK,) * 3
        cv2.circle(img, c, int(ring.radius_mm * PX), line, max(1, int(0.3 * PX)))
    if target.inner_ring_radius_mm:
        cv2.circle(img, c, int(target.inner_ring_radius_mm * PX), (PAPER,) * 3, max(1, int(0.3 * PX)))
    r = caliber_mm / 2 * PX
    for (x, y) in holes_mm:
        p = (int(c[0] + x * PX), int(c[1] + y * PX))
        cv2.circle(img, p, int(r), TAN, -1)  # torn rim
        cv2.circle(img, p, int(r * 0.7), DARK, -1)  # hole
    return img


def photograph(img, tmp_path):
    """Perspective warp onto a dark background, mild blur, saved as JPEG."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[80, 60], [w - 40, 30], [w - 70, h + 20], [40, h - 10]]) * 0.9
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (int(w * 0.95), int(h * 0.95)), borderValue=(60, 50, 40))
    out = cv2.GaussianBlur(out, (0, 0), 1.0)
    path = tmp_path / "card.jpg"
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


CASES = [
    ("nra_b2", 5.6, [(0, 0), (8.0, 9.0), (-40.0, 5.0), (0, -60.0)], [10, 10, 7, 5]),
    ("nra_b3", 5.6, [(2.0, 1.0), (-21.0, 0.0), (0, 45.0), (70.0, -30.0)], [10, 10, 8, 7]),
    ("issf_10m_air_pistol", 4.5, [(1.0, -1.0), (0, 14.0), (-30.0, 6.0)], [10, 9, 7]),  # last one cuts the 8 line, clear of the numeral
]


@pytest.mark.parametrize("name,caliber,holes,expected", CASES)
def test_synthetic_card_scores(tmp_path, name, caliber, holes, expected):
    target = load_target(name)
    path = photograph(render(target, holes, caliber), tmp_path)
    res = process_image(path, target, caliber)
    assert res.calibration.residual_mm < 0.3
    got = sorted(s.score.value for s in res.shots)
    assert got == sorted(expected), [(round(s.x_mm, 1), round(s.y_mm, 1), s.score.value) for s in res.shots]
    for (x, y) in holes:
        nearest = min(res.shots, key=lambda s: np.hypot(s.x_mm - x, s.y_mm - y))
        assert np.hypot(nearest.x_mm - x, nearest.y_mm - y) < 1.0
