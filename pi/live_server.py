#!/usr/bin/env python3
"""ScoreCam live service for the Pi: camera in, scored shots out, phone page on top.

Run on the Pi (from the project directory, in its venv):
    .venv/bin/python pi/live_server.py --target nra_b3 --caliber 5.6
Open on the phone:   http://scorecam.local:8000/

The camera streams a low-resolution picture continuously (phone preview and
motion watch). When the picture has been still for a few frames, one
full-resolution frame is analysed by scorecam.live.LiveScorer in a worker
thread; new holes become shot events, written to a session directory
(sessions/<id>/) and shown on the page.

HTTP:
    /                  the page
    /stream.mjpg       low-res MJPEG preview
    /status            JSON: state, series, shots so far, focus figure
    /events.json       JSON: all events of the current session
    /session/start?target=nra_b3&caliber=5.6&shooter=NAME   start a new session
    /session/stop
    /zoom              toggle the 1:1 centre crop in the preview (focusing aid)
    /capture           save a full-res still to ~/captures (as the viewfinder did)
"""
import argparse
import io
import json
import sys
import threading
import time
from datetime import datetime
from http import server
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorecam.live import LiveEvent, LiveScorer  # noqa: E402
from scorecam.session import SessionHeader, SessionWriter, ShotRecord  # noqa: E402
from scorecam.target import load_target  # noqa: E402

PORT = 8000
LORES = (1280, 960)
CAPTURE_DIR = Path.home() / "captures"
SESSION_ROOT = Path(__file__).resolve().parents[1] / "sessions"
FRAME_PERIOD = 0.2  # s between watched frames (5 fps is plenty for a still target)


PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>ScoreCam live</title>
<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}
img{width:100%;display:block}
.bar{display:flex;gap:6px;padding:6px}
button{flex:1;padding:12px;font-size:17px;border:0;border-radius:6px;background:#444;color:#eee}
button.on{background:#c80;color:#000;font-weight:bold}
#state{padding:4px 8px;font-size:20px;font-weight:bold}
#focus{padding:0 8px;color:#aaa}
table{width:100%;border-collapse:collapse;font-size:18px}
td,th{padding:4px 8px;text-align:left;border-bottom:1px solid #333}
td.s{font-weight:bold;font-size:22px}
#total{padding:8px;font-size:22px}</style></head><body>
<img src="/stream.mjpg">
<img id=tgt src="/target.jpg" alt="" style="display:none">
<div class=bar><button id=z onclick="go('/zoom')">Zoom OFF</button>
<button onclick="go('/capture')">Capture</button>
<button onclick="startSession()">New session</button></div>
<div id=state>starting</div><div id=focus></div>
<div id=total></div>
<table><thead><tr><th>#</th><th>series</th><th>score</th><th>where</th><th></th></tr></thead><tbody id=shots></tbody></table>
<script>
function go(u){fetch(u).then(r=>r.text()).then(t=>{if(/zoom/.test(t)){var z=document.getElementById('z');var on=/1:1/.test(t);z.textContent=on?'Zoom ON (1:1)':'Zoom OFF';z.className=on?'on':''}})}
function startSession(){var n=prompt('Shooter name (optional)','')||'';fetch('/session/start?shooter='+encodeURIComponent(n)).then(poll)}
var stamp=0;
function poll(){fetch('/status').then(r=>r.json()).then(s=>{
  if(s.stamp&&s.stamp!=stamp){stamp=s.stamp;var im=document.getElementById('tgt');im.src='/target.jpg?'+stamp;im.style.display='block'}
  document.getElementById('state').textContent=s.state+(s.series?'  series '+s.series:'')+(s.busy?'  analysing...':'');
  document.getElementById('focus').textContent='focus '+s.focus+'   '+s.target+' '+s.caliber+' mm   session '+s.session;
  var tb=document.getElementById('shots');tb.innerHTML='';var tot=0,x=0;
  s.shots.slice().reverse().forEach(function(h){tot+=h.value;if(h.inner)x++;
    var tr=document.createElement('tr');tr.innerHTML='<td>'+h.index+'</td><td>'+h.series+'</td><td class=s>'+h.value+(h.inner?'X':'')+' <small>'+h.decimal.toFixed(1)+'</small></td><td>'+h.clock+'</td><td><small>'+h.flags.join(' ')+'</small></td>';tb.appendChild(tr)});
  document.getElementById('total').textContent=s.shots.length?('total '+tot+' ('+x+'X) over '+s.shots.length+' shots'):'no shots yet';
}).catch(()=>{}).then(()=>setTimeout(poll,1000))}
poll();
</script></body></html>"""


def sharpness(gray):
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def clock(x, y):
    if np.hypot(x, y) < 1.0:
        return "centre"
    ang = (np.degrees(np.arctan2(x, -y)) + 360) % 360
    h = int(round(ang / 30)) % 12
    return f"{12 if h == 0 else h} o'clock"


class Service:
    def __init__(self, target_name: str, caliber: float):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.full = self.cam.sensor_resolution
        cfg = self.cam.create_video_configuration(
            main={"size": self.full, "format": "RGB888"},
            lores={"size": LORES, "format": "YUV420"},
            raw={"size": self.full},  # full-resolution readout: the analysed frame needs every pixel
            controls={"FrameDurationLimits": (200000, 200000)},  # 5 fps: a still target needs no more
            buffer_count=3)
        self.cam.configure(cfg)
        self.cam.start()
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.jpeg_cond = threading.Condition()
        self.zoomed = False
        self.focus = 0.0
        self.busy = False
        self.state = "starting"
        self.target_name, self.caliber = target_name, caliber
        self.scorer: LiveScorer | None = None
        self.writer: SessionWriter | None = None
        self.shots: list[dict] = []
        self.events: list[dict] = []
        self.target_jpeg: bytes | None = None
        self.stamp = 0
        self.start_session(target_name, caliber, "")
        threading.Thread(target=self.loop, daemon=True).start()

    # ---- session
    def start_session(self, target_name: str, caliber: float, shooter: str) -> str:
        with self.lock:
            if self.writer:
                self.writer.close()
            target = load_target(target_name)
            self.target_name, self.caliber = target.name, caliber
            self.scorer = LiveScorer(target, caliber)
            header = SessionHeader(id=datetime.now().strftime("%Y%m%d_%H%M%S"), target=target.name,
                                   caliber_mm=caliber, started=time.time(), shooter=shooter)
            self.writer = SessionWriter(SESSION_ROOT, header)
            self.shots, self.events = [], []
            self.state = "waiting for target"
            return header.id

    def stop_session(self):
        with self.lock:
            if self.writer:
                self.writer.close()
                self.writer = None
            self.state = "stopped"

    # ---- camera loop
    def loop(self):
        while True:
            t0 = time.time()
            req = self.cam.capture_request()
            try:
                yuv = req.make_array("lores")
                lores = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                full = None
                if self.scorer is not None and self.writer is not None and not self.busy:
                    ev = self.scorer.observe(lores, t0)
                    if ev is not None:
                        self.handle(ev, None)
                    if self.scorer.analysis_due:
                        full = cv2.cvtColor(req.make_array("main"), cv2.COLOR_RGB2BGR)
            finally:
                req.release()
            if full is not None:
                self.busy = True
                threading.Thread(target=self.analyse, args=(full, t0), daemon=True).start()
            # preview: the lores picture, or a 1:1 crop of its centre when zoomed (focus aid)
            if self.zoomed:
                h, w = lores.shape[:2]
                view = cv2.resize(lores[h // 2 - h // 6:h // 2 + h // 6, w // 2 - w // 6:w // 2 + w // 6], (w, h))
            else:
                view = lores
            g = cv2.cvtColor(lores, cv2.COLOR_BGR2GRAY)
            self.focus = sharpness(g[g.shape[0] // 2 - 100:g.shape[0] // 2 + 100, g.shape[1] // 2 - 150:g.shape[1] // 2 + 150])
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.jpeg_cond:
                    self.jpeg = buf.tobytes()
                    self.jpeg_cond.notify_all()
            time.sleep(max(0.0, FRAME_PERIOD - (time.time() - t0)))

    def analyse(self, full: np.ndarray, t: float):
        try:
            ev = self.scorer.analyse(full, t)
            if ev is not None:
                self.handle(ev, full)
        except Exception as e:  # keep the service alive; show the error on the page
            self.state = f"error: {e}"
        finally:
            self.busy = False

    def render_target(self, ev: LiveEvent):
        """The analysed frame with every hole marked: known ones white, this
        event's new shots yellow with their session number."""
        res = ev.result
        img = res.rectified.copy()
        T = res.T_mm_to_rect
        fresh = {id(s) for s in ev.shots}
        n0 = len(self.shots) - len(ev.shots) if ev.kind == "shots" else len(self.shots)
        k = 0
        for s in sorted(res.shots, key=lambda s: -s.score.decimal):
            if "off_target" in s.flags:
                continue
            p = (int(T[0, 2] + s.x_mm * T[0, 0]), int(T[1, 2] + s.y_mm * T[1, 1]))
            new = id(s) in fresh
            col = (0, 220, 255) if new else (255, 255, 255)
            cv2.circle(img, p, int(3.2 * T[0, 0]), col, 3)
            label = f"{s.score.value}{'X' if s.score.inner else ''}"
            if new:
                k += 1
                label = f"#{n0 + k} {label}"
            cv2.putText(img, label, (p[0] + 36, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        txt = {"locked": f"locked: {ev.existing} holes already on target", "shots": f"{len(ev.shots)} new",
               "still": "no change"}.get(ev.kind, ev.kind)
        cv2.putText(img, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3)
        scale = 900 / img.shape[1]
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.target_jpeg = buf.tobytes()
            self.stamp = int(time.time() * 1000)

    def handle(self, ev: LiveEvent, full: np.ndarray | None):
        with self.lock:
            if self.writer is None:
                return
            frame = self.writer.save_frame(full, ev.t) if (full is not None and ev.kind in ("locked", "shots")) else ""
            if ev.kind == "shots":
                recs = [ShotRecord(0, ev.series, ev.t, s.x_mm, s.y_mm, s.score.value, s.score.decimal,
                                   s.score.inner, list(s.flags), frame) for s in ev.shots]
                self.events.append(self.writer.shots(recs, ev.t))
                for r in recs:
                    self.shots.append({**r.__dict__, "clock": clock(r.x_mm, r.y_mm)})
                self.state = f"{len(ev.shots)} new shot{'s' if len(ev.shots) != 1 else ''}"
            elif ev.kind in ("locked", "lost"):
                self.events.append(self.writer.event(ev.kind, ev.t, series=ev.series, existing=ev.existing,
                                                     frame=frame, message=ev.message))
                self.state = ("locked, %d holes already on target" % ev.existing) if ev.kind == "locked" else "target lost"
            elif ev.kind == "moving":
                self.state = "moving"
            elif ev.kind == "still":
                self.state = "locked, no change" if self.scorer.locked else f"no target ({ev.message})"
            if ev.result is not None:
                self.render_target(ev)

    # ---- misc
    def toggle_zoom(self) -> str:
        self.zoomed = not self.zoomed
        return "zoom 1:1 centre" if self.zoomed else "zoom full frame"

    def capture(self) -> str:
        CAPTURE_DIR.mkdir(exist_ok=True)
        path = CAPTURE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        req = self.cam.capture_request()
        try:
            bgr = cv2.cvtColor(req.make_array("main"), cv2.COLOR_RGB2BGR)
        finally:
            req.release()
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return f"saved {path.name} ({bgr.shape[1]}x{bgr.shape[0]})"

    def status(self) -> dict:
        return {"state": self.state, "busy": self.busy, "series": self.scorer.series if self.scorer else 0,
                "locked": bool(self.scorer and self.scorer.locked), "focus": round(self.focus),
                "target": self.target_name, "caliber": self.caliber,
                "session": self.writer.header.id if self.writer else "none", "shots": self.shots,
                "stamp": self.stamp}


service: Service | None = None


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            self._send(PAGE, "text/html")
        elif u.path == "/status":
            self._send(json.dumps(service.status()), "application/json")
        elif u.path == "/events.json":
            self._send(json.dumps(service.events), "application/json")
        elif u.path == "/session/start":
            sid = service.start_session(q.get("target", service.target_name), float(q.get("caliber", service.caliber)),
                                        q.get("shooter", ""))
            self._send(json.dumps({"session": sid}), "application/json")
        elif u.path == "/session/stop":
            service.stop_session()
            self._send("stopped", "text/plain")
        elif u.path == "/target.jpg":
            if service.target_jpeg is None:
                self.send_error(404)
            else:
                self._send(service.target_jpeg, "image/jpeg")
        elif u.path == "/zoom":
            self._send(service.toggle_zoom(), "text/plain")
        elif u.path == "/capture":
            self._send(service.capture(), "text/plain")
        elif u.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with service.jpeg_cond:
                        service.jpeg_cond.wait()
                        frame = service.jpeg
                    if frame is None:
                        continue
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


class ThreadedServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="nra_b3")
    ap.add_argument("--caliber", type=float, default=5.6)
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    service = Service(a.target, a.caliber)
    print(f"ScoreCam live at http://scorecam.local:{a.port}/  target {a.target} caliber {a.caliber}  sessions -> {SESSION_ROOT}")
    ThreadedServer(("", a.port), Handler).serve_forever()
