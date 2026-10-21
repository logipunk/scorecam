import numpy as np

from scorecam.calibrate import _robust_smooth_circular
from scorecam.detect import fill_small_gaps


def test_fill_small_gaps_fills_hole_interior_not_ring_interior():
    m = np.zeros((200, 200), np.uint8)
    yy, xx = np.indices(m.shape)
    # a thin annulus (torn rim) with a small interior
    r = np.hypot(xx - 50, yy - 50)
    m[(r > 18) & (r < 24)] = 1
    # a big thin circle (printed ring line) with a large interior
    r2 = np.hypot(xx - 130, yy - 130)
    m[(r2 > 60) & (r2 < 62)] = 1
    out = fill_small_gaps(m, max_area_px=1500)
    assert out[50, 50] == 1  # small interior filled
    assert out[130, 130] == 0  # large interior untouched


def test_robust_smooth_drops_outliers_and_interpolates():
    x = np.zeros(180)
    x[10:30] = 2.0  # a hole-sized outlier run
    x[100:120] = np.nan  # nothing found here
    s = _robust_smooth_circular(x, half=30, outlier=0.4)
    assert np.all(np.abs(s) < 0.1)
    assert not np.isnan(s).any()


def test_robust_smooth_follows_slow_variation():
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    x = 0.8 * np.cos(ang)
    s = _robust_smooth_circular(x, half=30, outlier=0.4)
    assert np.allclose(s, x, atol=0.3)
