"""
╔══════════════════════════════════════════════════════════════════╗
║        FRUIT NINJA BOT v2  —  ROBUST + DYNAMIC WINDOW            ║
║              Optimized for i3 CPU + 8GB RAM                      ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT:  SendInput (Method B - Confirmed Working)                ║
║  WINDOW: Auto-detected BlueStacks App Player                     ║
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

MOUSEEVENTF_MOVE     = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE          = 0

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("iu", INPUT_UNION)
    ]

SW = user32.GetSystemMetrics(0)
SH = user32.GetSystemMetrics(1)

def to_abs_coords(x, y):
    return int(x * 65535 / SW), int(y * 65535 / SH)

def send_input_swipe(x1, y1, x2, y2, duration=0.035, steps=10):
    def send_mouse(x, y, flags):
        ax, ay = to_abs_coords(x, y)
        mi = MOUSEINPUT(ax, ay, 0, flags, 0, None)
        inp = INPUT(INPUT_MOUSE, INPUT_UNION(mi=mi))
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    # Start swipe
    send_mouse(x1, y1, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.005)

    # Interpolate movement
    delay = duration / steps
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(x1 + (x2 - x1) * t)
        iy = int(y1 + (y2 - y1) * t)
        send_mouse(ix, iy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        time.sleep(delay)

    # End swipe
    send_mouse(x2, y2, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP)


def get_window_rect(window_name="BlueStacks App Player"):
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return None, None, None, None, None

    # Get Client Rect
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
    CONF_THRESH : float = 0.50
    IOU_THRESH  : float = 0.45
    MAX_DET     : int   = 15
    SKIP_FRAMES : int   = 2

    SLICE_DURATION   : float = 0.035
    SLICE_STEPS      : int   = 10
    SLICE_PADDING    : int   = 20
    GROUP_DISTANCE   : int   = 120
    BOMB_SAFE_MARGIN : int   = 60
    SWIPE_COOLDOWN   : float = 0.02

    SHOW_DEBUG : bool = True
    SHOW_FPS   : bool = True

    def __post_init__(self):
        if self.FRUIT_CLASSES is None:
            self.FRUIT_CLASSES = [
                "apple", "watermelon", "orange", "banana",
                "kiwi", "pineapple", "mango", "peach",
            ]

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
        self.model = YOLO(model_path)
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
            x1 = max(0, g[0].x1 - cfg.SLICE_PADDING)
            x2 = min(win_w, g[0].x2 + cfg.SLICE_PADDING)
            y1 = y2 = g[0].cy
        else:
            x1, y1 = g[0].cx, g[0].cy
            x2, y2 = g[-1].cx, g[-1].cy
        
        # Convert to screen
        swipes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return swipes

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
        hwnd, self.ox, self.oy, self.w, self.h = get_window_rect()
        if not hwnd:
            raise RuntimeError("Could not find BlueStacks App Player")
        
        logging.info(f"Window found: {self.w}x{self.h} at ({self.ox}, {self.oy})")
        
        # 2. Setup DXCam
        region = (self.ox, self.oy, self.ox + self.w, self.oy + self.h)
        self.cap = DXCamCapture(region)
        
        # 3. Setup Model
        m_path = cfg.ONNX_PATH if cfg.USE_ONNX and Path(cfg.ONNX_PATH).exists() else cfg.MODEL_PATH
        self.inf = InferenceThread(m_path, cfg)
        
        self.dets = []
        self.swipes = []
        self.fn = 0

    def run(self):
        logging.info("Bot starting...")
        time.sleep(2)
        
        while True:
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
    bot = FruitNinjaBot()
    bot.run()
