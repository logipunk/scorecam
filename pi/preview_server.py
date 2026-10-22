#!/usr/bin/env python3
"""Phone viewfinder for the ScoreCam Pi.

Run on the Pi:   python3 pi/preview_server.py
Open on phone:   http://scorecam.local:8000/

Buttons: Zoom toggles a 1:1 centre crop (for focusing), Capture saves a
full-resolution JPEG to ~/captures/ and shows a sharpness figure for the
centre of the frame (higher = sharper; compare while turning the focus ring).
Only Picamera2 (preinstalled on Raspberry Pi OS) is needed.
"""

import io
import time
from datetime import datetime
from http import server
from pathlib import Path
from threading import Condition
from urllib.parse import urlparse

import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 8000
CAPTURE_DIR = Path.home() / "captures"
PREVIEW_SIZE = (1280, 960)


PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>ScoreCam viewfinder</title>
<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}
img{width:100%;display:block}
.bar{display:flex;gap:8px;padding:8px}
button{flex:1;padding:16px;font-size:20px;border:0;border-radius:6px;background:#444;color:#eee}
button.on{background:#c80;color:#000;font-weight:bold}
#msg{padding:0 8px 8px;min-height:1.4em;font-size:18px}</style></head><body>
<img src="/stream.mjpg">
<div class=bar><button id=z onclick="go('/zoom')">Zoom OFF (full frame)</button>
<button onclick="go('/capture')">Capture</button></div>
<div id=msg></div>
<div id=focus style="padding:0 8px 8px;font-size:28px;font-weight:bold">focus --</div>
<script>
function poll(){fetch('/sharpness').then(r=>r.text()).then(t=>{document.getElementById('focus').textContent='focus '+t;setTimeout(poll,500)}).catch(()=>setTimeout(poll,2000))}
poll();
function show(t){document.getElementById('msg').textContent=t;
  var z=document.getElementById('z'), on=/ON/.test(t)||/1:1/.test(t);
  if(/zoom/i.test(t)){z.textContent=on?'Zoom ON (1:1 centre)':'Zoom OFF (full frame)';z.className=on?'on':'';}}
function go(u){fetch(u).then(r=>r.text()).then(show)}
fetch('/status').then(r=>r.text()).then(show);
</script></body></html>"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def sharpness(gray):
    """Variance of the Laplacian; larger means more edge contrast."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


class Camera:
    def __init__(self):
        self.cam = Picamera2()
        self.full = self.cam.sensor_resolution
        self.preview_cfg = self.cam.create_video_configuration(main={"size": PREVIEW_SIZE})
        self.still_cfg = self.cam.create_still_configuration(main={"size": self.full})
        self.cam.configure(self.preview_cfg)
        self.output = StreamingOutput()
        self.zoomed = False
        self.cam.start_recording(MJPEGEncoder(), FileOutput(self.output))

    def toggle_zoom(self):
        w, h = self.full
        if self.zoomed:
            crop = (0, 0, w, h)
        else:
            cw, ch = PREVIEW_SIZE
            crop = ((w - cw) // 2, (h - ch) // 2, cw, ch)
        self.cam.set_controls({"ScalerCrop": crop})
        self.zoomed = not self.zoomed
        return "zoom 1:1 centre" if self.zoomed else "zoom full frame"

    def live_sharpness(self):
        """Sharpness of the centre of the live preview (same scale regardless of zoom)."""
        arr = self.cam.capture_array("main")
        h, w = arr.shape[:2]
        centre = arr[h // 2 - 150 : h // 2 + 150, w // 2 - 200 : w // 2 + 200]
        gray = centre.mean(axis=2) if centre.ndim == 3 else centre
        return sharpness(gray)

    def capture(self):
        CAPTURE_DIR.mkdir(exist_ok=True)
        path = CAPTURE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        self.cam.stop_recording()
        try:
            self.cam.configure(self.still_cfg)
            self.cam.start()
            time.sleep(0.5)  # let exposure settle after the mode switch
            request = self.cam.capture_request()
            try:
                request.save("main", str(path))
                arr = request.make_array("main")
            finally:
                request.release()
            self.cam.stop()
        finally:
            self.cam.configure(self.preview_cfg)
            self.cam.start_recording(MJPEGEncoder(), FileOutput(self.output))
            if self.zoomed:
                self.zoomed = False
                self.toggle_zoom()
        h, w = arr.shape[:2]
        centre = arr[h // 2 - 300 : h // 2 + 300, w // 2 - 300 : w // 2 + 300]
        gray = centre.mean(axis=2) if centre.ndim == 3 else centre
        return f"saved {path.name} ({w}x{h}) sharpness {sharpness(gray):.0f}"


camera = None


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _text(self, body, ctype="text/html"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._text(PAGE)
        elif path == "/zoom":
            self._text(camera.toggle_zoom(), "text/plain")
        elif path == "/sharpness":
            self._text(f"{camera.live_sharpness():.0f}", "text/plain")
        elif path == "/status":
            self._text("zoom 1:1 centre" if camera.zoomed else "zoom full frame", "text/plain")
        elif path == "/capture":
            self._text(camera.capture(), "text/plain")
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with camera.output.condition:
                        camera.output.condition.wait()
                        frame = camera.output.frame
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
    camera = Camera()
    time.sleep(1)
    print(f"viewfinder at http://scorecam.local:{PORT}/  captures -> {CAPTURE_DIR}")
    try:
        ThreadedServer(("", PORT), Handler).serve_forever()
    finally:
        camera.cam.stop_recording()
