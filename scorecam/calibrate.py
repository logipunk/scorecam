"""Per-image calibration: find the black disc, refine on all printed ring
circles, and return a mapping from image pixels to target millimetres.

The mapping is a homography (flat card seen in perspective) plus a smooth
polynomial correction that absorbs a bowed card or lens distortion. Circles only
constrain the radial direction, so the tangential part of the correction is
regularised towards zero.

Target frame: origin at the ring centre, x right, y down, units mm. Rotation about
the centre is not observable from circles, so the frame keeps the image's axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .target import Target

# One parameter set for all images.
DISC_SEARCH_SCALE = 4  # downscale factor for finding the black disc
DISC_MIN_FRAC = 0.04  # disc major axis must be at least this fraction of image width
DISC_MIN_FILL = 0.80  # blob area / fitted ellipse area
DISC_MIN_ASPECT = 0.45
DISC_SURROUND_CONTRAST = 1.6  # mean(surrounding paper) / mean(disc)
REFINE_ANGLE_STEP_DEG = 2
HOMOGRAPHY_WINDOWS_MM = (2.0, 1.2)  # radial search half-width per homography pass
POLY_WINDOWS_MM = (2.5, 1.2, 0.6)  # per polynomial-correction pass
POLY_KEEP_MM = (2.0, 0.8, 0.4)  # residual limit for points used in each pass
REFINE_MIN_CONTRAST = 0.12  # line/edge contrast relative to local range
RANSAC_THRESH_PX = 3.0
POLY_DEGREE = 3
POLY_RIDGE = 0.05  # regularisation of the (unobservable) tangential correction
POLY_INLIER_MM = 0.35  # points further than this from the model are dropped between passes
COORD_SCALE_MM = 70.0  # normalisation for polynomial monomials
WARP_FLAG_MM = 1.5  # flag the card when the polynomial correction exceeds this anywhere
LOCAL_FLAG_MM = 0.8  # ...or when a printed ring still sits this far from the model locally
FIT_FLAG_MM = 0.25  # flag the card when the ring residual RMS exceeds this
OFFSET_WINDOW_MM = 2.0  # search half-width when measuring where printed rings really are
OFFSET_SMOOTH_HALF = 30  # running-median half window, in angle samples (+-60 deg: wider than any hole)
OFFSET_OUTLIER_MM = 0.4  # samples further than this from the running median are dropped
OFFSET_MIN_SAMPLES = 10
OFFSET_MAX_MM = 1.5  # a smoothed offset beyond this is not a ring (obstruction, card edge): ignore it


@dataclass
class Calibration:
    H: np.ndarray  # 3x3, image px -> target mm (before polynomial correction)
    H_inv: np.ndarray
    poly: np.ndarray | None  # [2, n_monomials] correction coefficients, or None
    mm_per_px: float  # at the target centre
    residual_mm: float  # RMS ring-edge residual after correction
    n_points: int
    n_inliers: int
    disc_ellipse: tuple | None  # cv2 ellipse (center, axes, angle) in full-res px
    max_correction_mm: float = 0.0  # largest polynomial correction over the card
    flags: list[str] = field(default_factory=list)
    ring_radii_mm: np.ndarray | None = None  # model radius of each boundary
    ring_offsets_mm: np.ndarray | None = None  # measured - model, [n_rings, n_angles], NaN = not found
    ring_offsets_smooth: np.ndarray | None = None  # robustly smoothed, no NaN

    def ring_offset_at(self, r_mm, theta_rad) -> np.ndarray:
        """Local radial displacement (mm) of the printed rings near (r, theta): the
        ring the model puts at radius R is printed at R + offset. A hole scored
        against this cancels paper deformation, since hole and ring move together."""
        if self.ring_offsets_smooth is None:
            return np.zeros_like(np.asarray(r_mm, float))
        r_mm = np.atleast_1d(np.asarray(r_mm, float))
        theta = np.atleast_1d(np.asarray(theta_rad, float))
        n = self.ring_offsets_smooth.shape[1]
        a = (theta % (2 * np.pi)) / (2 * np.pi) * n
        i0 = np.floor(a).astype(int) % n
        fa = a - np.floor(a)
        out = np.empty(len(r_mm))
        for k in range(len(r_mm)):
            col = self.ring_offsets_smooth[:, i0[k]] * (1 - fa[k]) + self.ring_offsets_smooth[:, (i0[k] + 1) % n] * fa[k]
            out[k] = np.interp(r_mm[k], self.ring_radii_mm, col)  # clamps beyond the end rings
        return out

    def to_mm(self, pts_px) -> np.ndarray:
        m = _apply(self.H, pts_px)
        return m - _poly_eval(self.poly, m) if self.poly is not None else m

    def to_px(self, pts_mm) -> np.ndarray:
        m = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
        if self.poly is not None:
            # invert m_corr = m - Q(m) by two fixed-point steps
            q = m + _poly_eval(self.poly, m)
            q = m + _poly_eval(self.poly, q)
            m = q
        return _apply(self.H_inv, m)


class CalibrationError(RuntimeError):
    pass


def _apply(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ---------------------------------------------------------------- black disc
def find_black_disc(gray: np.ndarray) -> tuple:
    """Return the cv2 ellipse of the black disc's outer edge, in full-res px."""
    s = DISC_SEARCH_SCALE
    small = cv2.resize(gray, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 1.5)
    thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (blur < thresh).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= best_area or stats[i, cv2.CC_STAT_WIDTH] < DISC_MIN_FRAC * small.shape[1]:
            continue
        blob = _fill_holes((labels == i).astype(np.uint8))
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 20:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (a, b), ang = ellipse
        if min(a, b) <= 0 or min(a, b) / max(a, b) < DISC_MIN_ASPECT:
            continue
        if blob.sum() / (np.pi * a * b / 4) < DISC_MIN_FILL:
            continue
        if _surround_contrast(blur, blob, ellipse) < DISC_SURROUND_CONTRAST:
            continue
        best, best_area = ellipse, area
    if best is None:
        raise CalibrationError("black disc not found")
    (cx, cy), (a, b), ang = best
    return ((cx * s, cy * s), (a * s, b * s), ang)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = (mask * 255).astype(np.uint8)
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | (ff == 0).astype(np.uint8)


def _surround_contrast(gray: np.ndarray, blob: np.ndarray, ellipse: tuple) -> float:
    (cx, cy), (a, b), ang = ellipse
    ring = np.zeros_like(blob)
    cv2.ellipse(ring, ((cx, cy), (a * 1.3, b * 1.3), ang), 1, -1)
    cv2.ellipse(ring, ((cx, cy), (a * 1.08, b * 1.08), ang), 0, -1)
    if ring.sum() == 0:
        return 0.0
    return float(gray[ring > 0].mean() + 1) / float(gray[blob > 0].mean() + 1)


# ------------------------------------------------------- initial affine guess
def affine_from_ellipse(ellipse: tuple, radius_mm: float) -> np.ndarray:
    """Homography (affine) mapping the ellipse to a circle of radius_mm centred at 0,
    keeping the image's axis orientation."""
    (cx, cy), (w, h), ang = ellipse
    a, b = w / 2, h / 2
    th = np.radians(ang)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    S = np.diag([radius_mm / a, radius_mm / b])
    A = R.T @ S @ R
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = -A @ np.array([cx, cy])
    return H


# ------------------------------------------------------------- ring sampling
def _sample_profiles(gray: np.ndarray, to_px, radius_mm: float, window_mm: float, n_angles: int):
    """Sample the image along outward radial lines crossing the model circle.
    Returns (profiles [n_angles, n_samples], offsets_mm, model_pts_mm, px0, dirs, px_per_mm)."""
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radial = np.stack([np.cos(th), np.sin(th)], axis=1)
    model = radial * radius_mm
    px0 = to_px(model)
    dirs = to_px(model + radial * 0.1) - px0
    px_per_mm = np.linalg.norm(dirs, axis=1) / 0.1
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n_samples = int(2 * window_mm * np.median(px_per_mm)) | 1
    offs = np.linspace(-window_mm, window_mm, n_samples)
    pos = px0[:, None, :] + dirs[:, None, :] * (offs[None, :, None] * px_per_mm[:, None, None])
    prof = cv2.remap(gray, pos[..., 0].astype(np.float32), pos[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    return prof.astype(np.float32), offs, model, px0, dirs, px_per_mm


def _locate(profile: np.ndarray, kind: str) -> float | None:
    """Sub-sample index of the ring feature in a 1-D radial profile, or None."""
    if (profile < 0).any():
        return None  # left the image
    p = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 1.0).ravel()
    rng = p.max() - p.min()
    if rng < 1e-3:
        return None
    if kind == "dark_line":
        sig = -p
    elif kind == "light_line":
        sig = p
    else:
        g = np.gradient(p)
        sig = g if kind == "edge_out_bright" else -g
    k = int(np.argmax(sig))
    if k <= 0 or k >= len(sig) - 1:
        return None
    contrast = (sig[k] - np.median(sig)) / rng if kind.endswith("line") else sig[k] / rng
    if contrast < REFINE_MIN_CONTRAST:
        return None
    y0, y1, y2 = sig[k - 1], sig[k], sig[k + 1]
    denom = y0 - 2 * y1 + y2
    return k + (0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0)


def _collect(gray, to_px, target: Target, window_mm: float):
    """(model mm, found px) ring-edge correspondences over all boundaries."""
    n_angles = int(360 / REFINE_ANGLE_STEP_DEG)
    src, dst = [], []
    for b in target.boundaries():
        prof, offs, model, px0, dirs, ppm = _sample_profiles(gray, to_px, b.radius_mm, window_mm, n_angles)
        for i in range(n_angles):
            k = _locate(prof[i], b.kind)
            if k is None:
                continue
            off_mm = np.interp(k, np.arange(len(offs)), offs)
            src.append(model[i])
            dst.append(px0[i] + dirs[i] * off_mm * ppm[i])
    return np.array(src, np.float64).reshape(-1, 2), np.array(dst, np.float64).reshape(-1, 2)


# ------------------------------------------------------- polynomial correction
def _monomials(m: np.ndarray) -> np.ndarray:
    x, y = m[:, 0] / COORD_SCALE_MM, m[:, 1] / COORD_SCALE_MM
    cols = []
    for d in range(2, POLY_DEGREE + 1):  # degree >= 2 only: affine part lives in H
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.stack(cols, axis=1)


def _poly_eval(coef: np.ndarray, m: np.ndarray) -> np.ndarray:
    return _monomials(np.asarray(m, np.float64).reshape(-1, 2)) @ coef.T


def _fit_poly(src_mm: np.ndarray, found_mm: np.ndarray) -> np.ndarray:
    """Fit Q so that found - Q(found) ~ src, using only the radial component of the
    residual (the tangential one is unobservable) and ridge regularisation."""
    r_hat = src_mm / np.maximum(np.linalg.norm(src_mm, axis=1, keepdims=True), 1e-6)
    e_r = np.sum((found_mm - src_mm) * r_hat, axis=1)
    M = _monomials(found_mm)
    A = np.hstack([M * r_hat[:, :1], M * r_hat[:, 1:]])  # unknowns: [ax..., ay...]
    lam = POLY_RIDGE * np.sqrt(len(A))
    A_aug = np.vstack([A, lam * np.eye(A.shape[1])])
    b_aug = np.concatenate([e_r, np.zeros(A.shape[1])])
    sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return sol.reshape(2, -1)


def refine(gray: np.ndarray, H: np.ndarray, target: Target) -> Calibration:
    H_inv = np.linalg.inv(H)
    for window in HOMOGRAPHY_WINDOWS_MM:
        src, dst = _collect(gray, lambda p: _apply(H_inv, p), target, window)
        if len(src) < 30:
            raise CalibrationError(f"too few ring-edge points ({len(src)})")
        H_inv, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH_PX)
        if H_inv is None:
            raise CalibrationError("homography fit failed")
    H = np.linalg.inv(H_inv)
    cal = Calibration(H=H, H_inv=H_inv, poly=None, mm_per_px=0.0, residual_mm=0.0,
                      n_points=0, n_inliers=0, disc_ellipse=None)
    for window, keep_mm in zip(POLY_WINDOWS_MM, POLY_KEEP_MM):
        src, dst = _collect(gray, cal.to_px, target, window)
        found = _apply(H, dst)  # uncorrected mm
        # drop points the current model already puts far off (holes, numerals)
        err = np.linalg.norm(cal.to_mm(dst) - src, axis=1)
        keep = err <= keep_mm
        cal.poly = _fit_poly(src[keep], found[keep])
        err = np.linalg.norm(cal.to_mm(dst) - src, axis=1)
        inl = err <= POLY_INLIER_MM
        cal.n_points, cal.n_inliers = len(src), int(inl.sum())
        cal.residual_mm = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan")
    cal.mm_per_px = 1.0 / float(np.linalg.norm(cal.to_px([[1.0, 0.0]]) - cal.to_px([[0.0, 0.0]])))
    cal.max_correction_mm = _max_correction(cal, target)
    cal.ring_radii_mm, cal.ring_offsets_mm = measure_ring_offsets(gray, cal, target)
    cal.ring_offsets_smooth = np.stack([_robust_smooth_circular(row, OFFSET_SMOOTH_HALF, OFFSET_OUTLIER_MM)
                                        for row in cal.ring_offsets_mm])
    if cal.max_correction_mm > WARP_FLAG_MM or np.abs(cal.ring_offsets_smooth).max() > LOCAL_FLAG_MM:
        cal.flags.append("card_not_flat")
    if not cal.residual_mm < FIT_FLAG_MM:
        cal.flags.append("poor_ring_fit")
    return cal


def _max_correction(cal: Calibration, target: Target) -> float:
    r = target.outer_radius_mm
    g = np.linspace(-r, r, 41)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= r]
    return float(np.max(np.linalg.norm(_poly_eval(cal.poly, pts), axis=1)))


def measure_ring_offsets(gray: np.ndarray, cal: Calibration, target: Target,
                         window_mm: float = OFFSET_WINDOW_MM, exclude_mm: list | None = None,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Residual position of each printed ring per angle after correction (mm).
    NaN where no feature was found or where a known hole (exclude_mm: list of
    (x, y, radius) in mm) would corrupt the measurement."""
    n_angles = int(360 / REFINE_ANGLE_STEP_DEG)
    bounds = target.boundaries()
    radii = np.array([b.radius_mm for b in bounds])
    offsets = np.full((len(bounds), n_angles), np.nan)
    for j, b in enumerate(bounds):
        prof, offs, model, *_ = _sample_profiles(gray, cal.to_px, b.radius_mm, window_mm, n_angles)
        for i in range(n_angles):
            k = _locate(prof[i], b.kind)
            if k is not None:
                offsets[j, i] = np.interp(k, np.arange(len(offs)), offs)
        for (hx, hy, hr) in exclude_mm or []:
            near = np.hypot(model[:, 0] - hx, model[:, 1] - hy) < hr + window_mm + 0.3
            offsets[j, near] = np.nan
    return radii, offsets


def update_ring_offsets(cal: Calibration, gray: np.ndarray, target: Target, exclude_mm: list) -> None:
    """Re-measure the printed ring positions ignoring the given holes, and refresh
    the smoothed table used for scoring and the card flags."""
    cal.ring_radii_mm, cal.ring_offsets_mm = measure_ring_offsets(gray, cal, target, exclude_mm=exclude_mm)
    cal.ring_offsets_smooth = np.stack([_robust_smooth_circular(row, OFFSET_SMOOTH_HALF, OFFSET_OUTLIER_MM)
                                        for row in cal.ring_offsets_mm])
    flat = cal.max_correction_mm <= WARP_FLAG_MM and np.abs(cal.ring_offsets_smooth).max() <= LOCAL_FLAG_MM
    cal.flags = [f for f in cal.flags if f != "card_not_flat"] + ([] if flat else ["card_not_flat"])


def _robust_smooth_circular(x: np.ndarray, half: int, outlier: float) -> np.ndarray:
    """Running median on a circular series, dropping samples far from it, then
    linear interpolation across the gaps. Returns zeros if nothing is valid."""
    n = len(x)
    idx = (np.arange(-half, half + 1)[None, :] + np.arange(n)[:, None]) % n
    with np.errstate(all="ignore"):
        win = x[idx]
        med = np.nanmedian(win, axis=1) if not np.isnan(win).all() else np.full(n, np.nan)
        good = ~np.isnan(x) & (np.abs(x - med) <= outlier)
        win = np.where(good[idx], x[idx], np.nan)
        count = np.sum(~np.isnan(win), axis=1)
        med = np.full(n, np.nan)
        ok_rows = count >= OFFSET_MIN_SAMPLES
        if ok_rows.any():
            med[ok_rows] = np.nanmedian(win[ok_rows], axis=1)
    med[np.abs(med) > OFFSET_MAX_MM] = np.nan
    if np.isnan(med).all():
        return np.zeros(n)
    if np.isnan(med).any():
        ang = np.arange(n)
        ok = ~np.isnan(med)
        med = np.interp(ang, ang[ok], med[ok], period=n)
    return med


# ------------------------------------------------------------------- public
def calibrate(gray: np.ndarray, target: Target) -> Calibration:
    black = target.black_radii_mm
    if black is None:
        raise CalibrationError("calibration currently needs a target with a black disc")
    ellipse = find_black_disc(gray)
    H0 = affine_from_ellipse(ellipse, black[1])
    cal = refine(gray, H0, target)
    cal.disc_ellipse = ellipse
    return cal


def rectify(image: np.ndarray, cal: Calibration, target: Target, px_per_mm: float,
            margin_mm: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Warp the image into the target frame. Returns (rectified, T) where T maps
    target mm to rectified px and the image covers the whole card plus a margin."""
    w_mm, h_mm = target.card_mm
    ox, oy = target.center_offset_mm
    x0, y0 = -w_mm / 2 - ox - margin_mm, -h_mm / 2 - oy - margin_mm
    T = np.array([[px_per_mm, 0, -x0 * px_per_mm], [0, px_per_mm, -y0 * px_per_mm], [0, 0, 1]])
    W, Hh = int((w_mm + 2 * margin_mm) * px_per_mm), int((h_mm + 2 * margin_mm) * px_per_mm)
    yy, xx = np.indices((Hh, W), dtype=np.float64)
    mm = np.stack([x0 + xx.ravel() / px_per_mm, y0 + yy.ravel() / px_per_mm], axis=1)
    px = cal.to_px(mm)
    mapx = px[:, 0].reshape(Hh, W).astype(np.float32)
    mapy = px[:, 1].reshape(Hh, W).astype(np.float32)
    rect = cv2.remap(image, mapx, mapy, cv2.INTER_AREA if px_per_mm < 1 / cal.mm_per_px else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    return rect, T
