"""
╔══════════════════════════════════════════════════════════════════╗
║        FRUIT NINJA BOT v2  —  ROBUST + DYNAMIC WINDOW            ║
║              Optimized for i3 CPU + 8GB RAM                      ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT:  SendInput (Method B - Confirmed Working)                ║
║  WINDOW: Auto-detected Fruit Ninja (Google Play Games on PC)     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import dxcam
import threading
import queue
import time
import logging
import ctypes
import ctypes.wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════
#  WIN32 DEFS & DPI AWARENESS
# ══════════════════════════════════════════════════════════════════
try:
    # Set DPI Awareness to ensure coordinates match screen pixels
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

user32 = ctypes.windll.user32
_mouse_event = user32.mouse_event
_SetCursorPos = user32.SetCursorPos

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

SW = user32.GetSystemMetrics(0)
SH = user32.GetSystemMetrics(1)

def send_input_swipe(x1, y1, x2, y2, duration=0.035, steps=10):
    logging.info(f"Swiping from ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
    
    # Start swipe
    _SetCursorPos(int(x1), int(y1))
    time.sleep(0.008)
    _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01) # Wait slightly so game registers the press before we start dragging

    # Interpolate movement
    delay = duration / steps
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(x1 + (x2 - x1) * t)
        iy = int(y1 + (y2 - y1) * t)
        _SetCursorPos(ix, iy)
        time.sleep(delay)

    # End swipe
    time.sleep(0.01) # Wait slightly so game registers final position before release
    _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


# Known window titles used by Fruit Ninja on Google Play Games / direct install
_FRUIT_NINJA_TITLES = [
    "Fruit Ninja®",
    "Fruit Ninja",
    "Fruit Ninja Classic",
    "Fruit Ninja Classic®",
    "FruitNinja",
    "Play Games - game platform"
]

def find_fruit_ninja_window():
    """
    Enumerate all visible top-level windows and return the HWND of any
    whose title contains 'Fruit Ninja' or 'FruitNinja' (case-insensitive).
    """
    found = [0]
    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,   # HWND — must be c_void_p so it arrives as a plain int
        ctypes.c_void_p    # lParam
    )
    def _cb(hwnd, lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        title = buf.value.strip().lower()
        if "fruit ninja" in title or "fruitninja" in title:
            found[0] = hwnd
            return False
        return True
    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return found[0]


def get_window_rect():
    """
    Locate the Fruit Ninja window and return
    (hwnd, screen_x, screen_y, client_w, client_h).
    """
    # Fast path: try known exact titles
    hwnd = 0
    for title in _FRUIT_NINJA_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            logging.info(f"Window found via exact title: '{title}'")
            break
    # Slow path: enumerate all visible windows
    if not hwnd:
        hwnd = find_fruit_ninja_window()
        if hwnd:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            logging.info(f"Window found via enumeration: '{buf.value.strip()}'")
    if not hwnd:
        return None, None, None, None, None
    # Get Client Rect (excludes title bar / borders)
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    # Get Screen Origin
    pt = ctypes.wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return hwnd, pt.x, pt.y, w, h


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    MODEL_PATH : str  = r"C:\Fruit-Ninja\best.pt"
    ONNX_PATH  : str  = r"C:\Fruit-Ninja\best.onnx"
    USE_ONNX   : bool = True

    FRUIT_CLASSES : list = None
    BOMB_CLASS    : str  = "bomb"

    IMGSZ       : int   = 320
    CONF_THRESH : float = 0.35
    IOU_THRESH  : float = 0.45
    MAX_DET     : int   = 15
    SKIP_FRAMES : int   = 2

    SLICE_DURATION   : float = 0.05
    SLICE_STEPS      : int   = 15
    SLICE_PADDING    : int   = 50
    GROUP_DISTANCE   : int   = 120
    BOMB_SAFE_MARGIN : int   = 60
    SWIPE_COOLDOWN   : float = 0.05

    SHOW_DEBUG : bool = True
    SHOW_FPS   : bool = True

    def __post_init__(self):
        if self.FRUIT_CLASSES is None:
            self.FRUIT_CLASSES = ["fruit"]

CFG = Config()

# ══════════════════════════════════════════════════════════════════
#  THREADS
# ══════════════════════════════════════════════════════════════════
class DXCamCapture:
    def __init__(self, region: tuple):
        self._camera = dxcam.create(region=region, output_color="BGR")
        self._camera.start(target_fps=120, video_mode=True)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            f = self._camera.get_latest_frame()
            if f is not None:
                with self._lock: self._frame = f
            else:
                time.sleep(0.001)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._t.join(timeout=1)
        self._camera.stop()

class InferenceThread:
    def __init__(self, model_path, cfg):
        self.cfg = cfg
        self.model = YOLO(model_path, task='detect')
        self._in = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try: frame = self._in.get(timeout=0.5)
            except queue.Empty: continue

            results = self.model.predict(
                source=frame, imgsz=self.cfg.IMGSZ,
                conf=self.cfg.CONF_THRESH, iou=self.cfg.IOU_THRESH,
                max_det=self.cfg.MAX_DET, verbose=False
            )
            try: self._out.get_nowait()
            except queue.Empty: pass
            self._out.put(results[0])

    def submit(self, frame):
        try: self._in.put_nowait(frame)
        except queue.Full: pass

    def get_result(self):
        try: return self._out.get_nowait()
        except queue.Empty: return None

# ══════════════════════════════════════════════════════════════════
#  LOGIC
# ══════════════════════════════════════════════════════════════════
def plan_all(detections, cfg, ox, oy, win_w, win_h):
    bombs = [d for d in detections if d.cls == cfg.BOMB_CLASS]
    fruits = [d for d in detections if d.cls in cfg.FRUIT_CLASSES]
    
    # Filter safe fruits
    safe = []
    for f in fruits:
        too_close = False
        for b in bombs:
            dist = ((f.cx - b.cx)**2 + (f.cy - b.cy)**2)**0.5
            if dist < cfg.BOMB_SAFE_MARGIN:
                too_close = True; break
        if not too_close: safe.append(f)
    
    if not safe: return []

    # Simple clustering
    safe.sort(key=lambda d: d.cx)
    groups = []
    if safe:
        curr = [safe[0]]
        for i in range(1, len(safe)):
            if safe[i].cx - curr[-1].cx < cfg.GROUP_DISTANCE:
                curr.append(safe[i])
            else:
                groups.append(curr)
                curr = [safe[i]]
        groups.append(curr)

    swipes = []
    for g in groups:
        if len(g) == 1:
            x1 = max(0, g[0].cx - cfg.SLICE_PADDING)
            x2 = min(win_w, g[0].cx + cfg.SLICE_PADDING)
            y1 = y2 = g[0].cy
        else:
            x1, y1 = g[0].cx, g[0].cy
            x2, y2 = g[-1].cx, g[-1].cy
        
        # Convert to screen
        swipes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return swipes

class FruitTracker:
    def __init__(self):
        self.history = []
        self.last_time = time.time()
        self.predict_ahead = 0.15 # Predict 0.15s into the future
        
    def update(self, fruits):
        curr_time = time.time()
        dt = curr_time - self.last_time
        self.last_time = curr_time
        
        new_history = []
        for f in fruits:
            orig_cx, orig_cy = f.cx, f.cy
            
            # Only track if dt makes sense (prevent huge jumps if lag spikes)
            if 0.001 < dt < 0.2:
                best_dist = 150 
                best_prev = None
                for h in self.history:
                    dist = ((orig_cx - h['cx'])**2 + (orig_cy - h['cy'])**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_prev = h
                        
                if best_prev:
                    # Calculate velocity (pixels per second)
                    vx = (orig_cx - best_prev['cx']) / dt
                    vy = (orig_cy - best_prev['cy']) / dt
                    
                    # Offset the center coordinate forward in time
                    f.cx = int(orig_cx + vx * self.predict_ahead)
                    f.cy = int(orig_cy + vy * self.predict_ahead)
                    
            new_history.append({'cx': orig_cx, 'cy': orig_cy})
            
        self.history = new_history

class DetObj:
    __slots__ = ('x1','y1','x2','y2','cx','cy','cls','conf')
    def __init__(self, x1, y1, x2, y2, cls, conf):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cx, self.cy = (x1+x2)//2, (y1+y2)//2
        self.cls, self.conf = cls, conf

# ══════════════════════════════════════════════════════════════════
#  MAIN BOT
# ══════════════════════════════════════════════════════════════════
class FruitNinjaBot:
    def __init__(self, cfg=CFG):
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
        self.cfg = cfg
        
        # 1. Window Detect
        hwnd, orig_x, orig_y, self.w, self.h = get_window_rect()
        if not hwnd:
            raise RuntimeError(
                "Could not find Fruit Ninja window!\n"
                "Please launch Fruit Ninja via Google Play Games first, "
                "then re-run this script."
            )
        
        logging.info(f"Window found: {self.w}x{self.h} at ({orig_x}, {orig_y})")
        
        # Clamp region to screen bounds to prevent DXCam from crashing on negative coords
        left = max(0, orig_x)
        top = max(0, orig_y)
        right = min(SW, orig_x + self.w)
        bottom = min(SH, orig_y + self.h)
        
        if right <= left or bottom <= top:
            raise RuntimeError("The game window is completely off-screen!")
            
        # We use the clamped origin so the swipe coordinates map perfectly back to the desktop
        self.ox, self.oy = left, top
        
        # 2. Setup DXCam
        region = (left, top, right, bottom)
        self.cap = DXCamCapture(region)
        
        # 3. Setup Model
        m_path = cfg.ONNX_PATH if cfg.USE_ONNX and Path(cfg.ONNX_PATH).exists() else cfg.MODEL_PATH
        self.inf = InferenceThread(m_path, cfg)
        
        self.dets = []
        self.swipes = []
        self.fn = 0
        self.tracker = FruitTracker()

    def run(self):
        logging.info("Bot starting...")
        
        # Focus the Fruit Ninja window so it actually receives mouse events!
        hwnd, *_ = get_window_rect()
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            
        time.sleep(2)
        
        while True:
            # ── GLOBAL KILL SWITCH ──
            # 0x1B is the virtual key code for ESC.
            # GetAsyncKeyState checks the global keyboard state anywhere in Windows.
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                logging.info("\n[KILL SWITCH] ESC pressed! Stopping bot immediately.")
                break

            frame = self.cap.read()
            if frame is None: continue
            
            self.fn += 1
            if self.fn % self.cfg.SKIP_FRAMES == 0:
                self.inf.submit(frame)
            
            res = self.inf.get_result()
            if res is not None:
                self.dets = []
                for box in res.boxes:
                    coords = box.xyxy[0].tolist()
                    self.dets.append(DetObj(
                        int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]),
                        res.names[int(box.cls[0])], float(box.conf[0])
                    ))
                
                # Apply predictive physics tracking to fruits!
                fruits_only = [d for d in self.dets if d.cls in self.cfg.FRUIT_CLASSES]
                self.tracker.update(fruits_only)
                
                self.swipes = plan_all(self.dets, self.cfg, self.ox, self.oy, self.w, self.h)
                
                for s in self.swipes:
                    send_input_swipe(*s, duration=self.cfg.SLICE_DURATION, steps=self.cfg.SLICE_STEPS)
                    time.sleep(self.cfg.SWIPE_COOLDOWN)
            
            if self.cfg.SHOW_DEBUG:
                vis = frame.copy()
                for d in self.dets:
                    c = (0, 255, 0) if d.cls != 'bomb' else (0,0,255)
                    cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), c, 2)
                cv2.imshow("Bot Debug", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fruit Ninja Aim Bot (Predictive)")
    parser.add_argument('--no-debug', action='store_true', help='Hide the OpenCV debug window for better performance')
    args = parser.parse_args()
    
    # Create config and apply CLI overrides
    cfg = Config()
    if args.no_debug:
        cfg.SHOW_DEBUG = False

    bot = FruitNinjaBot(cfg)
    bot.run()
