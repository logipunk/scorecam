# ScoreCam

Live shot detection and scoring for 10 m air pistol and indoor bullseye.
See `CLAUDE.md` for the project overview and `docs/` for the brief.

## Phase 1: still-image prototype

```
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/evaluate.py            # all cards vs test-cards/ground_truth.yaml, images in out/
.venv/bin/python scripts/score_card.py test-cards/focus3.jpg --cues   # one card, plus cue maps
.venv/bin/python scripts/measure_target.py gamo_bone_collector_14cm test-cards/focus*.jpg
```

Targets (`targets/`): `nra_b2` and `nra_b3` (50 ft bullseye, 2020 NRA rulebook rules
4.2/4.3), `issf_10m_air_pistol` (NRA B-40), `gamo_bone_collector_14cm` (first test cards,
measured from photos). Caliber is a setting: `--caliber 5.6` for .22 LR, `11.5` for .45.

Layout: `scorecam/` library (`target`, `scoring`, `calibrate`, `detect`, `pipeline`, `annotate`),
`scripts/` CLI entry points, `tests/`, `targets/` target definitions (YAML data, no code).

Pipeline: find the black disc, fit all printed rings (homography plus a smooth
polynomial correction for bowed cards), rectify to 0.1 mm/px, detect holes from
lighting-relative cues, score by hole-edge distance against the printed rings as
measured on that card. Holes on numerals or in overlap groups are flagged for
manual review rather than silently decided.
