# Live Shot Scoring System — Project Brief

Open-source, web-based, live shot detection and scoring for 10m air pistol and indoor bullseye. Camera at the firing line, Raspberry Pi doing the work, phone or tablet as the display. Built for hobbyists and clubs, not ISSF certification.

**UI design canvas:** https://claude.ai/artifact/1G1XJsC3XGgDKK1t2JJ4go
(Phone match setup, phone live screen, desktop live screen. Clickable.)

---

## Goals

- See every shot live on a phone at the bench, with score, direction, and timing.
- Decimal scoring for 10m air pistol; integer + X for bullseye.
- Works fully offline (Pi hotspot, web app served from the Pi).
- Records session video, with each shot timestamped for replay.
- Cheap enough that other shooters can build one from a parts list.
- Open source: code, target library, printable mount, build docs.

## Disciplines

| Discipline | Distance | Target | Scoring |
|---|---|---|---|
| 10m air pistol | 10 m | ISSF 10m AP (10 ring 11.5 mm, rings +16 mm dia.) | Decimal (0.1 ring = 0.8 mm) |
| Indoor bullseye | 50 ft (~15 m) | NRA B-2 / B-3 | Integer + X |

**ISSF 10m AP rules (current):** 60 shots qualification, unlimited sighters in 15 min prep, integer scoring in qualification, decimals in finals. Major paper-target matches: 1 shot per card; lower-level matches: 2, 5 or 10 per card.

---

## Architecture

```
Canon EF-S 55-250 lens → Fotasy EF-to-C adapter → (C-CS ring) → Pi HQ Camera
  → Pi 4 (capture, detect, score, record, web server)
  → Pi Wi-Fi hotspot / WebSocket
  → Phone or tablet browser (live view, scores, replay)
```

- Camera on a tripod at the firing line, beside and below the shooter's sight line, off the bench (vibration).
- Nothing downrange (indoor ranges use target carriers).
- Optional later: radar trigger (shot timing + velocity), ArUco markers, video per shot.

### Detection pipeline (planned)

1. Calibrate: find rings (or markers) → homography → pixels to mm.
2. Frame differencing: new hole = new shot.
3. Hole center (sub-pixel fit) → distance to center − pellet/bullet radius → score.
4. Manual correction for overlaps, fliers, misses.
5. Card change: prompt or auto-detect fresh card → recalibrate.

### Optics math

- Pi HQ sensor: 6.29 mm wide, 4056 px.
- Target: ~0.1 mm/px for solid decimals.
- At ~200 mm focal length: 10 m → ~315 mm FOV (~0.08 mm/px); 50 ft → ~480 mm FOV (~0.12 mm/px).
- Air rifle decimals (0.025 mm) are out of reach for a camera; pistol is fine.

---

## Hardware

### Chosen (first build, manual focus)

| Part | Status | Cost |
|---|---|---|
| Raspberry Pi HQ Camera (C/CS mount) | Ordered | ~$55 |
| Canon EF-S 55-250 f/4-5.6 IS (original, non-STM), Adorama used | Buying | ~$90 delivered |
| Fotasy EF-to-C mount adapter (passive) | Buying | ~$20 |
| Raspberry Pi 4 (with PoE HAT + fan) | Have | — |
| Tripod (DX-TRP60) + flat bar mount | Have | — |
| USB-C power bank, 5V/3A, ~20,000 mAh | Needed | ~$30 |

Notes:
- Non-STM lens = mechanical focus ring, works on a passive adapter. Aperture stays wide open.
- Keep the C-CS spacer ring on the HQ camera, or the lens won't focus.
- Support the lens (390 g), not the camera board.
- Pi 4: HQ ribbon cable fits; hardware H.264 encoder helps recording + scoring together.
- Pi 4 power: ~5–7 W; 20,000 mAh bank ≈ 8–10 hours.

### Upgrade / alternative routes

| Route | Cost | Notes |
|---|---|---|
| Pinefeat EF controller + Canon 55-250 **STM** | ~$350–400 | Autofocus + aperture via libcamera. STM is focus-by-wire, needs power. |
| Camcorder (Canon Vixia HF R-series) + USB HDMI capture | ~$100–325 | Clean HDMI out, 32x optical, 1080p. Good "use what you own" path. |
| Spotting scope + digiscoping adapter | ~$20–60 if owned | Common in bullseye range bags. Alignment is fiddly. |
| Vintage M42 135–200 mm prime + adapter | ~$50–100 | Cheapest sharp option, manual everything. |

Ruled out: GoPro/webcams (too wide), UniFi G3-AF (fixed 3.6 mm), rifle scopes (reticle in image), clip-on phone "36x" lenses (soft).

### Mount

- First test: existing flat bar, hose clamps, Velcro.
- Later: 3D-printed mount (lens cradle, HQ behind it, Pi slot, tripod plate, power bank strap). Part of the open-source kit. Any working printer is fine; PLA or PETG.

---

## Software features

### MVP

- **Target library:** definition files (ring diameters, scoring zones, X-ring). ISSF 10m AP, NRA B-2, B-3 first.
- **Match setup:** presets (ISSF qualification, Club match, Training); shots per card 1/2/5/10; match shots; series size; time limit.
- **Card change:** "Pause and prompt" or "Auto-detect new card". **Next card** button always available.
- **Live screen:** SIUS-style layout. Target plot, last shot highlighted, shot list with direction arrows and decimal scores, totals, inner tens, group center (MPI), group size, splits.
- **Plot / Camera toggle:** clean plot, or live camera with markers overlaid for verification.
- **Shot editing:** move, delete, add (missed detection), miss (0 for off-card fliers).
- **Series and session:** series tabs, running totals, session history saved locally, CSV/PDF export.
- **Recording:** session video with shot timestamps, tap a shot to replay.
- **Offline:** Pi hotspot, no CDN dependencies, clock synced from the phone on connect (Pi 4 has no RTC).

### Later

- Trends across sessions.
- Radar trigger (exact timing, velocity).
- Multi-lane / club competition mode.
- Online leagues: honor system; club-verified sessions, lane photo, per-shot video. Can't fully prevent cheating; keep the data model ready.

---

## Existing work (reference, not reuse without license check)

- TargetSpotter (Android, closed): live, phone + spotting scope.
- freETarget (open source): acoustic electronic target.
- TargetScan, TargetIQ: photo-based scoring apps.
- GitHub: wrayx/target-score-cpp, Niv-Kor/Target-Score-Detector, sigmondkukla/Target-Analysis, Terranlee/Shooting-Target.
- SIUS: acoustic (older) and LS10 laser light curtain (in-plane, no camera).

Gap this project fills: live + camera at the firing line + web UI + offline + open source + DIY parts list.

---

## Next steps

1. Parts arrive → mount lens + adapter + HQ on the Pi 4.
2. Live view for focusing: `rpicam-vid -t 0 --inline --listen --width 1920 --height 1080 -o tcp://0.0.0.0:8888`, watch in VLC at `tcp/h264://<pi-ip>:8888`.
3. Full-res stills at 10 m and 50 ft, several zoom settings: `rpicam-still -o target_10m.jpg --width 4056 --height 3040` (add `-r` for RAW).
4. Check hole edge sharpness and mm/px from real stills.
5. Move to Claude Code: detection pipeline against real images first, then the web UI.

## Open questions

- License: MIT (adoption) vs GPL (improvements flow back).
- Project name.
- Hole visibility in the black under real range lighting.
- Tripod vibration at the bullseye line.
