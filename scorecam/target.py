"""Target definitions loaded from YAML data files (see targets/)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

TARGET_DIR = Path(__file__).resolve().parent.parent / "targets"


@dataclass(frozen=True)
class Ring:
    value: int
    radius_mm: float  # outer radius of this scoring zone
    color: str  # printed background: "white" or "black"


@dataclass(frozen=True)
class Boundary:
    """A printed circle on the card: a thin line or a black/white edge."""

    radius_mm: float
    kind: str  # "dark_line", "light_line", "edge_out_bright", "edge_out_dark"


@dataclass(frozen=True)
class Target:
    name: str
    description: str
    card_mm: tuple[float, float]
    center_offset_mm: tuple[float, float]
    rings: tuple[Ring, ...]  # ordered from the centre outward (highest value first)
    outer_color: str
    inner_ring_radius_mm: float | None
    numeral_angles_deg: tuple[float, ...] = ()  # where ring numerals are printed (0 = right, 90 = down)
    numeral_rings: tuple = ()  # ring values that carry a printed numeral; "X" = a mark at the centre
    ignore_regions_mm: tuple[tuple[float, float, float, float], ...] = ()  # printed text boxes/logos: x0,y0,x1,y1

    @property
    def outer_radius_mm(self) -> float:
        return self.rings[-1].radius_mm

    @property
    def black_radii_mm(self) -> tuple[float, float] | None:
        """(inner, outer) radius of the black disc, or None if the target has no black."""
        black = [r for r in self.rings if r.color == "black"]
        if not black:
            return None
        outer = max(r.radius_mm for r in black)
        inner_rings = [r for r in self.rings if r.radius_mm < min(b.radius_mm for b in black)]
        inner = max((r.radius_mm for r in inner_rings), default=0.0)
        return inner, outer

    def color_at(self, radius_mm: float) -> str:
        for ring in self.rings:
            if radius_mm <= ring.radius_mm:
                return ring.color
        return self.outer_color

    def boundaries(self) -> list[Boundary]:
        """Every printed circle, with how it looks when walking outward across it."""
        out = []
        inner_color = self.rings[0].color
        for i, ring in enumerate(self.rings):
            outer_color = self.rings[i + 1].color if i + 1 < len(self.rings) else self.outer_color
            out.append(Boundary(ring.radius_mm, _boundary_kind(inner_color, outer_color)))
            inner_color = outer_color
        return out


def _boundary_kind(inner: str, outer: str) -> str:
    if inner == outer:
        return "dark_line" if inner == "white" else "light_line"
    return "edge_out_dark" if outer == "black" else "edge_out_bright"


def load_target(name_or_path: str | Path) -> Target:
    path = Path(name_or_path)
    if not path.exists():
        path = TARGET_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        d = yaml.safe_load(f)
    rings = tuple(sorted((Ring(int(r["value"]), float(r["radius_mm"]), r["color"]) for r in d["rings"]),
                         key=lambda r: r.radius_mm))
    return Target(
        name=d["name"],
        description=d.get("description", ""),
        card_mm=tuple(float(v) for v in d["card_mm"]),
        center_offset_mm=tuple(float(v) for v in d.get("center_offset_mm", [0, 0])),
        rings=rings,
        outer_color=d.get("outer_color", "white"),
        inner_ring_radius_mm=d.get("inner_ring_radius_mm"),
        numeral_angles_deg=tuple(float(a) for a in d.get("numeral_angles_deg", [])),
        numeral_rings=tuple("X" if str(v).upper() == "X" else int(v) for v in d.get("numeral_rings", [])),
        ignore_regions_mm=tuple(tuple(float(v) for v in r) for r in d.get("ignore_regions_mm", [])),
    )
