"""Scoring math: ring lookup with the edge rule, and decimal scores."""
from __future__ import annotations

from dataclasses import dataclass

from .target import Target


@dataclass(frozen=True)
class Score:
    value: int  # integer ring value, 0 = miss / outside rings
    decimal: float  # e.g. 10.3; 0.0 for a miss
    inner: bool  # inside the inner ten / X ring
    distance_mm: float  # hole centre to target centre


def effective_distance(distance_mm: float, pellet_mm: float) -> float:
    """Distance of the hole's nearest edge from the centre (edge rule), floored at 0."""
    return max(0.0, distance_mm - pellet_mm / 2.0)


def ring_value(distance_mm: float, target: Target, pellet_mm: float) -> int:
    """Integer score. A hole touching a ring line scores the higher value."""
    d = effective_distance(distance_mm, pellet_mm)
    for ring in target.rings:  # innermost first
        if d <= ring.radius_mm:
            return ring.value
    return 0


def decimal_value(distance_mm: float, target: Target, pellet_mm: float) -> float:
    """Decimal score: each ring is split into ten equal-width zones (x.0 at the
    outer line, x.9 at the inner line); the innermost ring's inner edge is the centre."""
    d = effective_distance(distance_mm, pellet_mm)
    inner_r = 0.0
    for ring in target.rings:
        if d <= ring.radius_mm:
            width = ring.radius_mm - inner_r
            tenths = int((ring.radius_mm - d) / width * 10)
            return ring.value + min(tenths, 9) / 10.0
        inner_r = ring.radius_mm
    return 0.0


def score_hole(distance_mm: float, target: Target, pellet_mm: float = 4.5) -> Score:
    d = effective_distance(distance_mm, pellet_mm)
    inner = target.inner_ring_radius_mm is not None and d <= target.inner_ring_radius_mm
    return Score(
        value=ring_value(distance_mm, target, pellet_mm),
        decimal=decimal_value(distance_mm, target, pellet_mm),
        inner=inner,
        distance_mm=distance_mm,
    )
