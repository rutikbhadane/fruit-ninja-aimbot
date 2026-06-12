"""
╔══════════════════════════════════════════════════════════════════╗
║        FRUIT NINJA BOT (LEGACY STYLE) — MSS + PyAutoGUI          ║
║              Designed for Maximum Compatibility                  ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT:  PyAutoGUI (Slower but extremely reliable)               ║
║  CAPTURE: MSS (Solid fallback for DXCam issues)                  ║
║  WINDOW: Auto-detected Fruit Ninja (Google Play Games on PC)     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import mss
import pyautogui
import threading
import queue
import time
import logging
import ctypes
import ctypes.wintypes
from pathlib import Path
from ultralytics import YOLO

# Disable PyAutoGUI failsafe and delays for speed (careful!)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True 

# ══════════════════════════════════════════════════════════════════
#  DPI AWARENESS (Crucial for screen coordinates)
# ══════════════════════════════════════════════════════════════════
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

user32 = ctypes.windll.user32

_FRUIT_NINJA_TITLES = [
    "Fruit Ninja®",
    "Fruit Ninja",
    "Fruit Ninja Classic",
    "Fruit Ninja Classic®",
    "FruitNinja",
]

def _find_fruit_ninja_window():
    """Enumerate visible windows; return HWND matching 'fruit ninja' substring."""
    found = ctypes.c_void_p(0)
    EnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)
    )
    def _cb(hwnd, lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if "fruit ninja" in buf.value.strip().lower() or "fruitninja" in buf.value.strip().lower():
            found.value = hwnd
            return False
        return True
    user32.EnumWindows(EnumProc(_cb), 0)
    return found.value

def get_window_info():
    """Locate the Fruit Ninja window and return its geometry dict."""
    hwnd = 0
    for title in _FRUIT_NINJA_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            break
    if not hwnd:
        hwnd = _find_fruit_ninja_window()
    if not hwnd:
        return None
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    pt = ctypes.wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {"hwnd": hwnd, "x": pt.x, "y": pt.y, "w": w, "h": h}

class MSSCapture:
    def __init__(self, region):
        # region: (x, y, w, h)
        self.sct = mss.mss()
        self.monitor = {
            "top": region[1],
            "left": region[0],
            "width": region[2],
            "height": region[3]
        }
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            # Capture and convert to BGR
            img = np.array(self.sct.grab(self.monitor))
            self.frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # time.sleep(0.001)

    def read(self):
        return self.frame

class InferenceThread:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self._in = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            frame = self._in.get()
            results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=False)
            try: self._out.get_nowait()
            except queue.Empty: pass
            self._out.put(results[0])

    def submit(self, frame):
        try: self._in.put_nowait(frame)
        except queue.Full: pass

    def get_result(self):
        try: return self._out.get_nowait()
        except queue.Empty: return None

def run_bot():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    
    info = get_window_info()
    if not info:
        logging.error("Could not find Fruit Ninja window! Launch Fruit Ninja first.")
        return

    logging.info(f"Window detected at ({info['x']}, {info['y']}) size {info['w']}x{info['h']}")
    
    # 1. Start Capture
    cap = MSSCapture((info['x'], info['y'], info['w'], info['h']))
    
    # 2. Start Inference
    model_path = r"C:\Fruit-Ninja\best.pt"
    if Path(r"C:\Fruit-Ninja\best.onnx").exists():
        model_path = r"C:\Fruit-Ninja\best.onnx"
    inf = InferenceThread(model_path)
    
    logging.info("Bot logic running. Press Ctrl+C in terminal to stop.")
    time.sleep(2)

    last_dets = []
    
    while True:
        frame = cap.read()
        if frame is None:
            continue
        
        inf.submit(frame)
        res = inf.get_result()
        
        if res:
            dets = []
            for box in res.boxes:
                # [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                label = res.names[int(box.cls[0])]
                dets.append({
                    "x1": coords[0], "y1": coords[1],
                    "x2": coords[2], "y2": coords[3],
                    "label": label
                })
            last_dets = dets
            
            # Simple swipe logic
            fruits = [d for d in dets if d['label'] != 'bomb']
            bombs = [d for d in dets if d['label'] == 'bomb']
            
            # Group fruits for a single swipe if they are close
            if fruits:
                # Just slice the first fruit for testing simplicity
                f = fruits[0]
                cx, cy = (f['x1'] + f['x2']) / 2, (f['y1'] + f['y2']) / 2
                
                # Screen Absolute Coords
                screen_x = info['x'] + cx
                screen_y = info['y'] + cy
                
                logging.info(f"Slicing {f['label']} at ({screen_x}, {screen_y})")
                
                # THE PYAUTOGUI SLICE
                # Move to start of fruit, press down, move across, lift up
                pyautogui.moveTo(screen_x - 30, screen_y, _pause=False)
                pyautogui.dragTo(screen_x + 30, screen_y, duration=0.04, _pause=False)
        
        # Optional: Show debug window
        # for d in last_dets:
        #     cv2.rectangle(frame, (int(d['x1']), int(d['y1'])), (int(d['x2']), int(d['y2'])), (0,255,0), 2)
        # cv2.imshow("Debug", frame)
        # if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    run_bot()
