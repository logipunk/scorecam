import math

import pytest

from scorecam.scoring import decimal_value, effective_distance, ring_value, score_hole
from scorecam.target import load_target

GAMO = load_target("gamo_bone_collector_14cm")
ISSF = load_target("issf_10m_air_pistol")


def test_target_loads_sorted_rings():
    assert [r.value for r in GAMO.rings] == list(range(10, 0, -1))
    assert GAMO.black_radii_mm == (12.95, 33.15)
    assert GAMO.color_at(20.0) == "black"
    assert GAMO.color_at(5.0) == "white"
    assert GAMO.color_at(100.0) == "white"


def test_boundary_kinds():
    kinds = [b.kind for b in GAMO.boundaries()]
    assert kinds[:3] == ["dark_line", "edge_out_dark", "light_line"]
    assert kinds[4] == "edge_out_bright"  # 6/5 line: black -> white
    assert kinds[-1] == "dark_line"


def test_effective_distance_floor():
    assert effective_distance(1.0, 4.5) == 0.0
    assert effective_distance(10.0, 4.5) == pytest.approx(7.75)


def test_centre_shot_is_ten():
    assert ring_value(0.0, GAMO, 4.5) == 10
    assert decimal_value(0.0, GAMO, 4.5) == 10.9


def test_edge_rule_scores_higher_ring():
    r10 = GAMO.rings[0].radius_mm  # 6.35
    # centre just outside the 10 ring but the hole edge still touches it -> 10
    assert ring_value(r10 + 2.25 - 0.01, GAMO, 4.5) == 10
    # hole edge clear of the line -> 9
    assert ring_value(r10 + 2.25 + 0.01, GAMO, 4.5) == 9


def test_miss_outside_rings():
    assert ring_value(GAMO.outer_radius_mm + 2.25 + 0.1, GAMO, 4.5) == 0
    assert decimal_value(GAMO.outer_radius_mm + 2.25 + 0.1, GAMO, 4.5) == 0.0
    assert ring_value(GAMO.outer_radius_mm + 2.25 - 0.1, GAMO, 4.5) == 1


def test_decimal_issf_values():
    # ISSF: 9 ring is 5.75..13.75 mm (8 mm wide, 0.8 mm per tenth)
    assert decimal_value(5.75 + 2.25 + 0.1, ISSF, 4.5) == pytest.approx(9.9)
    assert decimal_value(13.75 + 2.25 - 0.1, ISSF, 4.5) == pytest.approx(9.0)
    assert decimal_value(5.75 + 2.25 + 4.0, ISSF, 4.5) == pytest.approx(9.5)


def test_caliber_setting_changes_edge_rule():
    d = GAMO.rings[0].radius_mm + 2.5  # a 5.5 mm pellet still touches the 10 ring, a 4.5 mm one does not
    assert ring_value(d, GAMO, 5.5) == 10
    assert ring_value(d, GAMO, 4.5) == 9


def test_inner_ten():
    assert score_hole(1.0, ISSF).inner is True
    assert score_hole(6.0, ISSF).inner is False
    assert score_hole(1.0, GAMO).inner is False


def test_score_monotonic_in_distance():
    last = math.inf
    for d in [x * 0.5 for x in range(0, 160)]:
        s = decimal_value(d, GAMO, 4.5)
        assert s <= last
        last = s
