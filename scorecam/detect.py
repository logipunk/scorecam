"""Hole detection on a rectified (mm-space) card image.

The target model says which zones are printed white and which black. Within each
zone the expected intensity is the local paper (white zones) or ink (black zones)
level, estimated with morphology using a disc that fits inside one ring band but
not inside a pellet hole. Cues, all relative so lighting does not matter:
  dark   = darker than local paper in a white zone      (hole on white)
  bright = brighter than local ink in a black zone      (torn rim in the black)
  warm   = chroma shifted towards tan torn-paper fibre  (rim, pale holes)
The union is cleaned with morphology sized by the pellet diameter: fill small
enclosed gaps (hole interiors), then open with a disc that is wider than any
printed numeral stroke or line but narrower than a pellet hole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import Calibration
from .target import Target

# One parameter set for all images (all sizes in mm).
BG_DISC_MM = 5.5  # local background disc: > hole diameter, < ring band width
EDGE_BAND_MM = 0.5  # ignore this band either side of a black/white edge (alignment)
DARK_FRAC = 0.09  # hole must be this fraction darker than local paper
BRIGHT_FRAC = 0.20  # rim must be this fraction (of paper level) brighter than local ink (halo of dented paper is ~0.1-0.18)
WARM_SHIFT = 12.0  # Lab (a+b) shift from the zone's own tint
FILL_MAX_MM2 = 12.0  # enclosed gaps smaller than this are filled
OPEN_DISC_MM = 2.6  # opening disc: kills lines/strokes, keeps a hole core (a .22 hole in paper closes to ~3 mm)
RECOVER_MM = 0.6  # grow kept blobs back into the raw mask by this much (torn edges, not line stubs)
MIN_DIAMETER_MM = 3.2  # single-hole equivalent diameter range
MAX_DIAMETER_MM = 6.2
MIN_ROUNDNESS = 0.45  # 4*pi*area/perimeter^2; a torn .22 hole in paper is ragged, ~0.5
MIN_SOLIDITY = 0.60  # blob area / convex hull area (overlap groups too)
MAX_PAPER_FRAC = 0.20  # blob pixels at paper level: torn holes have few, numerals many
PAPER_LEVEL_FRAC_WHITE = 0.92  # "paper level" in white zones, relative to local paper
PAPER_LEVEL_FRAC_BLACK = 0.80  # white numeral strokes in the black blur a little darker
NUMERAL_RADIUS_MM = 3.5  # a candidate centred within this of a printed numeral is suspect
NUMERAL_PATCH_MM = 7.5  # patch around a numeral position in which printed strokes are removed
NUMERAL_STROKE_MAX_MM = 1.4  # printed structures thinner than this are glyph strokes, not a hole core
NUMERAL_ALIGN_MM = 0.5  # sibling patches may be shifted this much when registering them
WARM_EDGE_GRAD = 18.0  # grey levels per px: warm cue is ignored near stronger edges (colour fringing)
WARM_EDGE_GROW_MM = 0.3
NUMERAL_STROKE_GROW_MM = 0.35  # grow the stroke template before subtracting it from a blob
MIN_NUMERAL_EXTRA_MM2 = 3.0  # blob area beyond the printed strokes needed to call it a hole
NUMERAL_EXTRA_OPEN_MM = 2.0  # opening applied to that extra area (drops edge slivers, keeps a hole core)
INK_LEVEL_FRAC = 0.55  # "ink level" in white zones: darker than this fraction of local paper
LINE_BAND_MM = 0.6  # half-width of the band blanked around each thin printed ring line
BRIDGE_MM = 1.4  # closing that rejoins a solid hole cut by a blanked line band
ZONES_FOLLOW_RINGS = False  # place zone edges at measured ring positions (worse on obstructed cards)
MAX_SINGLE_LENGTH_FACTOR = 1.45  # a blob longer than this x the pellet AND elongated holds more than one
MAX_SINGLE_ASPECT = 1.4  # length / width above which a long blob is an overlap group
CARD_EDGE_MARGIN_MM = 2.5


@dataclass
class Hole:
    x_mm: float  # rectified-image px (converted to mm by the caller)
    y_mm: float
    diameter_mm: float
    roundness: float
    area_mm2: float
    flags: list[str] = field(default_factory=list)  # "overlap"
    n_est: int = 1  # estimated number of pellets in this blob
    paper_frac: float = 0.0  # fraction of blob pixels at printed-paper level
    warm_frac: float = 0.0  # fraction of blob pixels with torn-paper colour
    solidity: float = 1.0


@dataclass
class DetectionDebug:
    dark: np.ndarray
    bright: np.ndarray
    warm: np.ndarray
    candidates: np.ndarray
    card_mask: np.ndarray
    numeral_strokes: np.ndarray
    rejected: list[dict]


def _disc(diameter_mm: float, px_per_mm: float) -> np.ndarray:
    d = max(3, int(round(diameter_mm * px_per_mm)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def zone_masks(shape: tuple, px_per_mm: float, T: np.ndarray, target: Target, cal: Calibration | None = None,
               polar: tuple | None = None):
    """(white_zone, black_zone) boolean masks from the target model, with every
    black/white edge placed where it was measured on this card (if `cal` carries
    ring offsets) and a band around it excluded from both zones."""
    r, theta = polar if polar is not None else polar_maps(shape, px_per_mm, T)
    is_black = np.zeros(r.shape, bool)
    edge = np.zeros(r.shape, bool)
    inner_r = 0.0
    for ring, b in zip(target.rings, target.boundaries()):
        outer_r = b.radius_mm
        if ring.color == "black":
            is_black |= (r >= inner_r) & (r < outer_r)
        if b.kind.startswith("edge"):
            if cal is not None and cal.ring_offsets_smooth is not None:
                near = np.abs(r - outer_r) < EDGE_BAND_MM + float(np.abs(cal.ring_offsets_smooth).max()) + 0.1
                local = local_ring_radius(outer_r, theta[near], cal)
                edge[near] = np.abs(r[near] - local) < EDGE_BAND_MM
                blk = np.zeros(r.shape, bool); blk[near] = r[near] < local
                is_black[near] = blk[near] if ring.color == "black" else ~blk[near] & is_black[near]
            else:
                edge |= np.abs(r - outer_r) < EDGE_BAND_MM
        inner_r = outer_r
    if target.outer_color == "black":
        is_black |= r >= inner_r
    return (~is_black) & ~edge, is_black & ~edge


def local_ring_radius(radius_mm: float, theta: np.ndarray, cal: Calibration | None) -> np.ndarray:
    """Radius of the printed ring that the model puts at radius_mm, per pixel angle."""
    if cal is None or cal.ring_offsets_smooth is None:
        return np.full(theta.shape, radius_mm, np.float32)
    k = int(np.argmin(np.abs(cal.ring_radii_mm - radius_mm)))
    off = cal.ring_offsets_smooth[k]
    n = len(off)
    a = (theta % (2 * np.pi)) / (2 * np.pi) * n
    i0 = np.floor(a).astype(int) % n
    fa = a - np.floor(a)
    return (radius_mm + off[i0] * (1 - fa) + off[(i0 + 1) % n] * fa).astype(np.float32)


def find_card_mask(bgr: np.ndarray, px_per_mm: float, T: np.ndarray, target: Target,
                   ) -> tuple[np.ndarray, float]:
    """Card area in the rectified image, as a rotated rectangle fitted to the bright
    region around the ring centre, shrunk by a margin so the card edge, and
    anything leaning on it, is never a candidate. Also returns the card's rotation
    (radians) relative to the rectified axes."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.3 * px_per_mm)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (blur > t).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, _disc(2.0, px_per_mm))
    cx, cy = int(T[0, 2]), int(T[1, 2])
    cv2.circle(bright, (cx, cy), int(target.outer_radius_mm * px_per_mm), 1, -1)  # bridge the black
    n, labels = cv2.connectedComponents(bright)
    card = (labels == labels[cy, cx]).astype(np.uint8)
    theta = _card_rotation(card, px_per_mm, (cx, cy), target)
    # rotate so the card is axis aligned, then take robust row/column extents
    R = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), 1.0)
    upright = cv2.warpAffine(card, R, card.shape[::-1], flags=cv2.INTER_NEAREST)
    w_mm, h_mm = target.card_mm
    band = int(0.25 * min(w_mm, h_mm) * px_per_mm)
    rows = upright[cy - band:cy + band]
    cols = upright[:, cx - band:cx + band]
    left, right = _extents(rows, axis=1)
    top, bottom = _extents(cols.T, axis=1)
    # never trust an extent beyond the card's own size (leaves, white backgrounds)
    left = max(left, cx - (w_mm / 2 + 3) * px_per_mm)
    right = min(right, cx + (w_mm / 2 + 3) * px_per_mm)
    top = max(top, cy - (h_mm / 2 + 3) * px_per_mm)
    bottom = min(bottom, cy + (h_mm / 2 + 3) * px_per_mm)
    m = CARD_EDGE_MARGIN_MM * px_per_mm
    rect = np.array([[left + m, top + m], [right - m, top + m], [right - m, bottom - m], [left + m, bottom - m]])
    Rinv = cv2.invertAffineTransform(R)
    pts = (Rinv @ np.hstack([rect, np.ones((4, 1))]).T).T
    mask = np.zeros_like(card)
    cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1)
    return mask, theta


def mask_ignore_regions(card: np.ndarray, px_per_mm: float, T: np.ndarray, target: Target,
                        theta_card: float) -> np.ndarray:
    """Blank printed logos and text boxes listed in the target file (mm rectangles
    relative to the ring centre, in the card's own orientation)."""
    out = card.copy()
    c, s_ = np.cos(theta_card), np.sin(theta_card)
    for (x0, y0, x1, y1) in target.ignore_regions_mm:
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        rot = corners @ np.array([[c, s_], [-s_, c]])  # rotate with the card
        pts = np.round(np.array([T[0, 2], T[1, 2]]) + rot * px_per_mm).astype(np.int32)
        cv2.fillConvexPoly(out, pts, 0)
    return out


def _extents(strip: np.ndarray, axis: int) -> tuple[float, float]:
    """Median first/last foreground index across the lines of a strip."""
    has = strip.any(axis=axis)
    if not has.any():
        return 0.0, float(strip.shape[axis])
    first = np.argmax(strip[has], axis=axis)
    last = strip.shape[axis] - 1 - np.argmax(strip[has][:, ::-1], axis=axis)
    return float(np.median(first)), float(np.median(last))


def _card_rotation(card: np.ndarray, px_per_mm: float, centre: tuple, target: Target) -> float:
    """Rotation of the card edges relative to the image axes (radians, mod 90 deg)."""
    edges = cv2.morphologyEx(card, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    yy, xx = np.indices(card.shape)
    r = np.hypot(xx - centre[0], yy - centre[1]) / px_per_mm
    half = min(target.card_mm) / 2
    edges[(r < half * 0.9) | (r > half * 1.5)] = 0  # keep only the card outline band
    lines = cv2.HoughLinesP(edges * 255, 1, np.pi / 360, int(10 * px_per_mm),
                            minLineLength=int(20 * px_per_mm), maxLineGap=int(3 * px_per_mm))
    if lines is None:
        return 0.0
    angs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = np.arctan2(y2 - y1, x2 - x1) % (np.pi / 2)
        angs.append(a)
        weights.append(np.hypot(x2 - x1, y2 - y1))
    angs = np.array(angs)
    angs[angs > np.pi / 4] -= np.pi / 2  # -45..45 deg
    order = np.argsort(angs)
    cum = np.cumsum(np.array(weights)[order])
    return float(angs[order][np.searchsorted(cum, cum[-1] / 2)])  # weighted median


def polar_maps(shape: tuple, px_per_mm: float, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (radius_mm, angle) about the ring centre."""
    h, w = shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    dx, dy = xx - T[0, 2], yy - T[1, 2]
    return np.hypot(dx, dy) / px_per_mm, np.arctan2(dy, dx)


def thin_line_band(r: np.ndarray, theta: np.ndarray, target: Target, cal: Calibration | None) -> np.ndarray:
    """Pixels within LINE_BAND_MM of a thin printed ring line, at its measured
    position. The measured radius is evaluated only inside a narrow annulus
    around each line (the offsets are bounded), which is what makes this cheap."""
    band = np.zeros(r.shape, bool)
    reach = LINE_BAND_MM + (float(np.abs(cal.ring_offsets_smooth).max()) if cal is not None and cal.ring_offsets_smooth is not None else 0.0)
    for b in target.boundaries():
        if not b.kind.endswith("line"):
            continue
        near = np.abs(r - b.radius_mm) < reach + 0.1
        local = local_ring_radius(b.radius_mm, theta[near], cal)
        hit = np.abs(r[near] - local) < LINE_BAND_MM
        band[near] = hit
    return band


def numeral_centres(target: Target, theta_card: float, cal: Calibration | None) -> list[tuple[float, float]]:
    """Target-frame (mm) centres of the printed ring numerals."""
    out = []
    if "X" in target.numeral_rings:  # a printed X (or other mark) at the centre
        out.append((0.0, 0.0))
    inner = target.inner_ring_radius_mm or 0.0
    for ring in target.rings:
        # the innermost ring's numeral (e.g. the "10" on an NRA B-2) sits at the centre
        mid = 0.0 if inner == 0.0 else (inner + ring.radius_mm) / 2
        inner = ring.radius_mm
        if ring.value not in target.numeral_rings:
            continue
        for a in target.numeral_angles_deg:
            th = np.radians(a) + theta_card
            rr = mid + (float(cal.ring_offset_at(mid, th)[0]) if cal is not None else 0.0)
            out.append((rr * np.cos(th), rr * np.sin(th)))
    return out


def numeral_stroke_mask(printed: np.ndarray, px_per_mm: float, T: np.ndarray, target: Target,
                        theta_card: float, cal: Calibration | None, card_mask: np.ndarray) -> np.ndarray:
    """Printed strokes of the ring numerals, to be subtracted from a candidate that
    sits on a numeral. Two sources, united: (1) within a patch around each numeral
    position, the printed structures thinner than a stroke width (a hole has a
    solid core wider than any stroke); (2) when numerals are printed at three or
    more angles, the strokes the sibling glyphs agree on (handles soft focus, where
    strokes blur wide)."""
    half = int(NUMERAL_PATCH_MM / 2 * px_per_mm)
    thick = cv2.morphologyEx(printed, cv2.MORPH_OPEN, _disc(NUMERAL_STROKE_MAX_MM, px_per_mm))
    thin = printed & ~thick
    out = np.zeros(printed.shape, np.uint8)
    for (x, y) in numeral_centres(target, theta_card, cal):
        cx, cy = int(round(T[0, 2] + x * px_per_mm)), int(round(T[1, 2] + y * px_per_mm))
        y0, y1, x0, x1 = max(cy - half, 0), cy + half, max(cx - half, 0), cx + half
        out[y0:y1, x0:x1] |= thin[y0:y1, x0:x1]
    out |= numeral_sibling_strokes(printed, px_per_mm, T, target, theta_card, cal, card_mask)
    return cv2.dilate(out, _disc(2 * NUMERAL_STROKE_GROW_MM, px_per_mm))


def numeral_sibling_strokes(printed: np.ndarray, px_per_mm: float, T: np.ndarray, target: Target,
                        theta_card: float, cal: Calibration | None, card_mask: np.ndarray) -> np.ndarray:
    """Printed strokes of the ring numerals, learned from the card itself: glyphs
    are upright at every angle, so the sibling patches of one numeral agree on
    the strokes (majority vote after registration) while a hole, or the ring arcs
    that rotate with the angle, appear in one patch only."""
    out = np.zeros(printed.shape, np.uint8)
    half = int(NUMERAL_PATCH_MM / 2 * px_per_mm)
    pad = int(NUMERAL_ALIGN_MM * px_per_mm)
    centres = numeral_centres(target, theta_card, cal)
    n_ang = len(target.numeral_angles_deg)
    if n_ang < 3:
        return out
    for i in range(0, len(centres), n_ang):
        patches, boxes = [], []
        for (x, y) in centres[i:i + n_ang]:
            cx, cy = int(round(T[0, 2] + x * px_per_mm)), int(round(T[1, 2] + y * px_per_mm))
            y0, y1, x0, x1 = cy - half - pad, cy + half + pad, cx - half - pad, cx + half + pad
            if y0 < 0 or x0 < 0 or y1 > printed.shape[0] or x1 > printed.shape[1]:
                continue
            if card_mask[y0:y1, x0:x1].mean() < 0.99:
                continue
            patches.append(printed[y0:y1, x0:x1].astype(np.float32))
            boxes.append((y0, x0))
        if len(patches) < 3:
            continue
        cores = [p[pad:-pad, pad:-pad] for p in patches]
        template = (np.sum(cores, axis=0) >= max(2, len(cores) - 1)).astype(np.float32)
        if template.sum() < 10:
            continue
        shifts = []
        for p in patches:  # register each sibling to the first vote (normalised: shape, not mass), vote again
            res = cv2.matchTemplate(p, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, loc = cv2.minMaxLoc(res)
            shifts.append(loc)
        aligned = [p[dy:dy + 2 * half, dx:dx + 2 * half] for p, (dx, dy) in zip(patches, shifts)]
        template = (np.sum(aligned, axis=0) >= max(2, len(aligned) - 1)).astype(np.uint8)
        for (y0, x0), (dx, dy) in zip(boxes, shifts):
            out[y0 + dy:y0 + dy + 2 * half, x0 + dx:x0 + dx + 2 * half] |= template
    return out



def cue_maps(bgr: np.ndarray, px_per_mm: float, white_zone: np.ndarray, black_zone: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.08 * px_per_mm)
    se = _disc(BG_DISC_MM, px_per_mm)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se).astype(np.float32)  # local white level
    ink = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se).astype(np.float32)  # local dark level
    g = gray.astype(np.float32)
    paper_level = float(np.median(paper[white_zone])) if white_zone.any() else float(gray.max())
    # paper level is only meaningful in white zones; inside the black use the card-wide level
    paper = np.where(white_zone, paper, paper_level).astype(np.float32)
    dark = ((paper - g) > DARK_FRAC * paper) & white_zone
    bright = ((g - ink) > BRIGHT_FRAC * paper_level) & black_zone
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    warm_raw = lab[..., 1].astype(np.float32) + lab[..., 2].astype(np.float32)
    warm_raw = cv2.GaussianBlur(warm_raw, (0, 0), 0.15 * px_per_mm)
    warm = np.zeros(gray.shape, bool)
    for zone in (white_zone, black_zone):
        if zone.any():
            warm |= ((warm_raw - float(np.median(warm_raw[zone]))) > WARM_SHIFT) & zone
    # lens colour fringing makes every strong edge look warm: ignore warm there
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    strong = (np.hypot(gx, gy) > WARM_EDGE_GRAD).astype(np.uint8)
    warm &= cv2.dilate(strong, _disc(2 * WARM_EDGE_GROW_MM, px_per_mm)) == 0
    at_paper = (g > PAPER_LEVEL_FRAC_WHITE * paper) & white_zone
    at_paper |= (g > PAPER_LEVEL_FRAC_BLACK * paper_level) & black_zone
    # printed marks (numeral strokes): ink level on white, the bright cue in the black
    printed = ((g < INK_LEVEL_FRAC * paper) & white_zone) | (bright & black_zone)
    return dark, bright, warm, at_paper, printed, paper_level


def fill_small_gaps(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    small = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] <= max_area_px and stats[i, cv2.CC_STAT_LEFT] > 0
             and stats[i, cv2.CC_STAT_TOP] > 0
             and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < w
             and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < h]
    if small:
        out[np.isin(labels, small)] = 1
    return out


def detect_holes(bgr: np.ndarray, px_per_mm: float, T: np.ndarray, target: Target,
                 cal: Calibration | None = None, pellet_mm: float = 4.5,
                 ) -> tuple[list[Hole], DetectionDebug]:
    card, theta_card = find_card_mask(bgr, px_per_mm, T, target)
    card = mask_ignore_regions(card, px_per_mm, T, target, theta_card)
    polar = polar_maps(bgr.shape, px_per_mm, T)
    white_zone, black_zone = zone_masks(bgr.shape, px_per_mm, T, target, cal if ZONES_FOLLOW_RINGS else None, polar)
    white_zone &= card > 0
    black_zone &= card > 0
    dark, bright, warm, at_paper, printed, paper_level = cue_maps(bgr, px_per_mm, white_zone, black_zone)
    r_map, th_map = polar
    band = thin_line_band(r_map, th_map, target, cal)
    # On white, blank the printed ring lines so they cannot chain numerals and holes together.
    # In the black a torn rim is itself a thin ring: blanking a line through it cuts it and a
    # closing cannot reliably rejoin it, so leave the white lines in; the opening below removes them.
    cue = (bright | ((dark | warm) & ~band)).astype(np.uint8)
    cue = cv2.morphologyEx(cue, cv2.MORPH_CLOSE, _disc(BRIDGE_MM, px_per_mm))  # across the blanked band
    filled = fill_small_gaps(cue, FILL_MAX_MM2 * px_per_mm ** 2)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, _disc(OPEN_DISC_MM, px_per_mm))
    # grow the surviving blobs back into the raw mask a little to recover torn edges
    cands = filled & cv2.dilate(opened, _disc(2 * RECOVER_MM, px_per_mm))
    strokes = numeral_stroke_mask(printed.astype(np.uint8), px_per_mm, T, target, theta_card, cal, card)
    numerals = numeral_centres(target, theta_card, cal)
    single_area = np.pi * (pellet_mm / 2) ** 2
    holes, rejected = [], []
    cnts, _ = cv2.findContours(cands, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        blob = np.zeros(cands.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 1, -1)
        shape = _blob_shape(c, px_per_mm)
        paper_frac = float(at_paper[blob > 0].mean())
        warm_frac = float(warm[blob > 0].mean())
        info = dict(**shape, paper_frac=paper_frac, warm_frac=warm_frac)
        if shape["diameter"] < MIN_DIAMETER_MM:
            rejected.append({**info, "why": "too small"})
            continue
        flags = []
        x_mm = (shape["x"] - T[0, 2]) / px_per_mm
        y_mm = (shape["y"] - T[1, 2]) / px_per_mm
        if any(np.hypot(x_mm - nx, y_mm - ny) <= NUMERAL_RADIUS_MM for nx, ny in numerals):
            # on a printed numeral: what is left after removing the printed strokes is the hole
            extra = (blob & ~strokes).astype(np.uint8)
            core = cv2.morphologyEx(extra, cv2.MORPH_OPEN, _disc(NUMERAL_EXTRA_OPEN_MM, px_per_mm))
            core_area = core.sum() / px_per_mm ** 2
            if core_area < MIN_NUMERAL_EXTRA_MM2:
                rejected.append({**info, "why": f"numeral ({core_area:.1f} mm2 extra)"})
                continue
            flags.append("on_numeral")
            hole = extra & cv2.dilate(core, _disc(2 * RECOVER_MM, px_per_mm))
            hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, _disc(2 * NUMERAL_STROKE_GROW_MM + 1.0, px_per_mm))
            e_cnts, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(e_cnts, key=cv2.contourArea)
            shape = _blob_shape(c, px_per_mm)
            info = dict(**shape, paper_frac=paper_frac, warm_frac=warm_frac)
        elif paper_frac > MAX_PAPER_FRAC:
            rejected.append({**info, "why": f"printed ({paper_frac:.2f} paper)"})
            continue
        if shape["solidity"] < MIN_SOLIDITY:
            rejected.append({**info, "why": f"not solid ({shape['solidity']:.2f})"})
            continue
        common = dict(paper_frac=paper_frac, warm_frac=warm_frac, solidity=shape["solidity"])
        multi = (shape["length"] > MAX_SINGLE_LENGTH_FACTOR * pellet_mm
                 and shape["length"] / max(shape["width"], 0.1) > MAX_SINGLE_ASPECT)
        if not multi and shape["diameter"] <= MAX_DIAMETER_MM + 1.0:
            if shape["roundness"] < MIN_ROUNDNESS:
                rejected.append({**info, "why": f"not round ({shape['roundness']:.2f})"})
                continue
            holes.append(Hole(shape["x"], shape["y"], shape["diameter"], shape["roundness"], shape["area"],
                              flags=flags, **common))
        elif shape["length"] <= 4 * pellet_mm:
            # how many pellets: no more than the blob's length allows, no more than its area allows
            n_est = max(2, min(4, int(round(shape["area"] / (single_area * 1.1))), int(round(shape["length"] / pellet_mm))))
            # longer than one pellet: an overlap group (best effort split, flagged)
            for (hx, hy) in split_overlap(c, n_est, pellet_mm, px_per_mm):
                holes.append(Hole(hx, hy, pellet_mm, shape["roundness"], shape["area"],
                                  flags=flags + ["overlap"], n_est=n_est, **common))
        else:
            rejected.append({**info, "why": f"too big (d={shape['diameter']:.1f})"})
    return holes, DetectionDebug(dark, bright, warm, cands, card, strokes, rejected)


def _blob_shape(c: np.ndarray, px_per_mm: float) -> dict:
    area_px = max(cv2.contourArea(c), 1.0)
    area = area_px / px_per_mm ** 2
    perim = cv2.arcLength(c, True) / px_per_mm
    hull_area = cv2.contourArea(cv2.convexHull(c)) / px_per_mm ** 2
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else float(c[:, 0, 0].mean())
    cy = M["m01"] / M["m00"] if M["m00"] else float(c[:, 0, 1].mean())
    (_, (rw, rh), _) = cv2.minAreaRect(c)
    return dict(x=cx, y=cy, area=area, diameter=2 * np.sqrt(area / np.pi), length=max(rw, rh) / px_per_mm,
                width=min(rw, rh) / px_per_mm,
                roundness=4 * np.pi * area / perim ** 2 if perim > 0 else 0.0,
                solidity=area / hull_area if hull_area > 0 else 0.0)


def split_overlap(contour: np.ndarray, n: int, pellet_mm: float, px_per_mm: float,
                  ) -> list[tuple[float, float]]:
    """Best-effort split of a merged blob into n pellet-sized discs using the
    distance transform: take the deepest point, suppress one pellet radius around
    it, repeat. Results are flagged; manual correction is expected."""
    x, y, w, h = cv2.boundingRect(contour)
    sub = np.zeros((h + 2, w + 2), np.uint8)
    cv2.drawContours(sub, [contour - [x - 1, y - 1]], -1, 1, -1)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    r_px = pellet_mm / 2 * px_per_mm
    centres = []
    for _ in range(n):
        k = int(np.argmax(dist))
        py, px = divmod(k, dist.shape[1])
        if dist[py, px] <= 0:
            break
        centres.append((float(px + x - 1), float(py + y - 1)))
        cv2.circle(dist, (px, py), int(r_px * 0.9), 0, -1)
    return centres
