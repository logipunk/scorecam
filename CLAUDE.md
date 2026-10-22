# ScoreCam — Live Shot Scoring System

Open-source, web-based, live shot detection and scoring for 10m air pistol and indoor bullseye.
Camera at the firing line, Raspberry Pi does the work, phone/tablet browser is the display.
For hobbyists and clubs, not ISSF certification.

## Two hardware pieces, one software platform
1. **Scoring camera (current focus)**: Pi 4 + HQ camera + telephoto lens on a tripod at the
   firing line. Sees real holes in paper and scores live fire.
2. **Dry-fire trainer ("SCATT-style", later phase)**: small sensor clamped to the pistol barrel.
   Tracks the aiming black at 100+ fps to produce an aim trace (hold, wobble, drift through the
   break). Works dry fire (no pellet) and live fire.
   - Hardware: see Parts below. Target weight ~55-70 g (SCATT MX-W2 is 56 g).
   - Target: printed black disk under a lamp (>=1000 lux), scaled for shooting distance.
   - Aim point = (disk center - calibrated reference) x mm/px, rotated for cant.
     Calibration = one or more dry/live shots with the shooter's normal sight picture.
   - Optional later: read shot events from the Morini CM162EI Bluetooth trigger board
     (protocol undocumented; reverse-engineer, read-only). IMU stays the baseline.
   - Live-fire calibration can use the scoring camera's detected holes automatically.

Shared across both: web UI, target library (data files), session/series model, offline Pi hotspot.
Design data models (shots, timestamps, sessions, target definitions) so trainer traces can
attach to the same sessions as scored shots later. Do not build trainer code until asked.

## Current phase
Phase 1 (scoring camera): hole detection + scoring prototype against real still images (test-cards/).
Web UI and live camera pipeline come after detection is proven on stills.

## Environment
- Dev machine: Pop!_OS, Python 3.12. Develop and test here first.
- Target device: Raspberry Pi 4 (Raspberry Pi OS Trixie, 64-bit), hostname `scorecam`.
- Camera: Raspberry Pi HQ Camera (IMX477, 4056x3040) + Canon EF-S 55-250 IS (non-STM) via passive EF-to-C adapter. Manual focus, aperture wide open.
- Capture on Pi: `rpicam-still`, `rpicam-vid` (libcamera / Picamera2).

## Test data
- `test-cards/` — full-res JPGs (4056x3040) of shot Gamo 14x14 cm cards.
  Shot at ~9.2 m, ~200-250 mm focal length, lamp lighting. ~0.06-0.07 mm/px.
- `test-cards/ground_truth.yaml` — per card: file, target, distance, focal length, hand scores per hole (0 = flier/miss).
- Gamo card is NOT the ISSF target. Ring dimensions for it go in the target definition (to be measured).
- Hard cases included: holes in the black, touching/overlapping holes, line-touchers, fliers outside rings, a pale hole on white, strong card tilt.

## Scoring rules
- 10m air pistol: ISSF 10m AP target (10 ring 11.5 mm dia, each ring +16 mm dia, black = rings 7-10, 59.5 mm). Decimal scoring possible (0.1 ring = 0.8 mm).
- Bullseye: NRA B-2 / B-3, integer + X.
- Edge scoring: a hole touching a ring line scores the higher value. Use hole center distance minus pellet/bullet radius.
- Pellet 4.5 mm. Caliber is a setting.

## Detection approach (planned)
1. Calibrate: find card rings (or ArUco markers later) -> homography -> pixel-to-mm.
2. Live mode: frame differencing between stable frames (new hole = new shot), with per-frame alignment.
3. Still mode (Phase 1): detect holes directly on single images.
4. Hole center via sub-pixel fit; overlap handling; manual correction always available.
5. Bullseye ranges use turning targets: the target faces the shooter only during the string
   ("fire" = target turns to face). Live mode must gate on "rings locked" (face-on) and compare
   the first stable face-on frame of a string with the last one before the turn; each face-on
   period gets its own calibration since the target never returns to exactly the same angle.
- Observed on real NRA sheets: a .22 hole closes to a ~3 mm dark core with torn fibres (bright
  C-shaped rim in the black). Fresh holes with a backstop behind look darker and cleaner.
- Classical CV (OpenCV) first. ML only where classical fails (e.g. overlaps).

## Architecture (target)
Pi does capture, detection, scoring, recording, and serves the web app.
Phone connects to the Pi's own Wi-Fi hotspot. Fully offline: no CDN, no cloud dependencies.
Target definitions are data files (rings, zones, X-ring) so new targets can be added without code.

## Docs
- `docs/shot-scoring-project-brief.md` — full project brief (hardware, optics math, decisions).
- `docs/` mockup HTML — UI design: match setup, phone live screen, desktop live screen.

## Conventions
- Keep it simple and testable. Scripts must run headless.
- Every detection change gets checked against ground_truth.yaml; report accuracy per card.
- Don't add dependencies that won't install on a Pi 4.
- Open source, MIT licensed (LICENSE).