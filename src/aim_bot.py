"""
╔══════════════════════════════════════════════════════════════════╗
║        FRUIT NINJA BOT  —  YOLOv8 ONNX + DXCam + Win32         ║
║              Optimized for i3 CPU + 8GB RAM                     ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT:  SetCursorPos + mouse_event (main thread only)          ║
║  WINDOW: BlueStacks 5 — 664x407 at desktop origin (68, 23)      ║
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════
#  INPUT — SetCursorPos + mouse_event (main thread only)
#  SetForegroundWindow and mouse_event are silently ignored when
#  called from background threads on Windows — so swipes MUST
#  execute in the main thread. This is why the cursor didn't move.
# ══════════════════════════════════════════════════════════════════
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

_user32        = ctypes.windll.user32
_mouse_event   = _user32.mouse_event
_SetCursorPos  = _user32.SetCursorPos
_SetForeground = _user32.SetForegroundWindow


def focus_bluestacks(hwnd: int):
    """Must be called from main thread to take effect."""
    _SetForeground(hwnd)
    time.sleep(0.05)


def win32_swipe(x1: int, y1: int, x2: int, y2: int,
                duration: float, steps: int):
    """
    Smooth swipe from (x1,y1) → (x2,y2).
    x1,y1,x2,y2 are desktop-absolute pixel coordinates.
    Must be called from main thread.
    """
    _SetCursorPos(x1, y1)
    time.sleep(0.008)
    _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.005)

    delay = duration / steps
    for i in range(1, steps + 1):
        t  = i / steps
        ix = int(x1 + (x2 - x1) * t)
        iy = int(y1 + (y2 - y1) * t)
        _SetCursorPos(ix, iy)
        time.sleep(delay)

    _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
@dataclass
class Config:

    # ── Model ─────────────────────────────────────────────────────
    MODEL_PATH : str  = "best.pt"
    USE_ONNX   : bool = True
    ONNX_PATH  : str  = "best.onnx"

    # ── Class names (fill from your data.yaml) ────────────────────
    FRUIT_CLASSES : list = None
    BOMB_CLASS    : str  = "bomb"

    def __post_init__(self):
        if self.FRUIT_CLASSES is None:
            self.FRUIT_CLASSES = ["fruit"]

    # ── Game window (confirmed via GetClientRect + ClientToScreen) ─
    #    Client size:   664 x 407
    #    Client origin: (68, 23)
    #    dxcam region:  (68, 23, 732, 430)
    # ──────────────────────────────────────────────────────────────
    CLIENT_W        : int   = 664
    CLIENT_H        : int   = 407
    CLIENT_ORIGIN_X : int   = 68
    CLIENT_ORIGIN_Y : int   = 23
    CAPTURE_REGION  : tuple = (68, 23, 732, 430)

    # ── Inference ─────────────────────────────────────────────────
    IMGSZ       : int   = 320
    CONF_THRESH : float = 0.50
    IOU_THRESH  : float = 0.45
    MAX_DET     : int   = 15
    SKIP_FRAMES : int   = 2

    # ── Swipe ─────────────────────────────────────────────────────
    SLICE_DURATION   : float = 0.035
    SLICE_STEPS      : int   = 12
    SLICE_PADDING    : int   = 20
    GROUP_DISTANCE   : int   = 100
    BOMB_SAFE_MARGIN : int   = 55
    SWIPE_COOLDOWN   : float = 0.03

    # ── Debug ─────────────────────────────────────────────────────
    SHOW_DEBUG : bool = True
    SHOW_FPS   : bool = True


CFG = Config()


# ══════════════════════════════════════════════════════════════════
#  ONNX EXPORT — one-time, then cached
# ══════════════════════════════════════════════════════════════════
def ensure_onnx(cfg: Config) -> str:
    if not cfg.USE_ONNX:
        return cfg.MODEL_PATH

    onnx_file = Path(cfg.ONNX_PATH)
    if onnx_file.exists():
        logging.info(f"[ONNX] Using cached model → {onnx_file}")
        return str(onnx_file)

    logging.info("[ONNX] Exporting (one-time, ~30s)...")
    model = YOLO(cfg.MODEL_PATH)
    model.export(format="onnx", imgsz=cfg.IMGSZ, simplify=True, opset=12)

    auto = Path(cfg.MODEL_PATH).with_suffix(".onnx")
    if auto.exists() and auto != onnx_file:
        auto.rename(onnx_file)

    logging.info(f"[ONNX] Done → {onnx_file}")
    return str(onnx_file)


# ══════════════════════════════════════════════════════════════════
#  THREAD 1 — DXCam Capture
# ══════════════════════════════════════════════════════════════════
class DXCamCapture:
    def __init__(self, region: tuple, target_fps: int = 120):
        self._camera = dxcam.create(region=region, output_color="BGR")
        self._camera.start(target_fps=target_fps, video_mode=True)
        self._frame   : Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        logging.info(f"[DXCam] Started — region:{region} @ {target_fps}fps")

    def _loop(self):
        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        self._t.join(timeout=2)
        self._camera.stop()
        logging.info("[DXCam] Stopped.")


# ══════════════════════════════════════════════════════════════════
#  THREAD 2 — YOLO Inference
# ══════════════════════════════════════════════════════════════════
class InferenceThread:
    def __init__(self, model_path: str, cfg: Config):
        self.cfg      = cfg
        self.model    = YOLO(model_path)
        self._in      = queue.Queue(maxsize=1)
        self._out     = queue.Queue(maxsize=1)
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        logging.info(f"[Inference] Ready — {model_path}")

    def _loop(self):
        while self._running:
            try:
                frame = self._in.get(timeout=0.5)
            except queue.Empty:
                continue

            results = self.model.predict(
                source  = frame,
                imgsz   = self.cfg.IMGSZ,
                conf    = self.cfg.CONF_THRESH,
                iou     = self.cfg.IOU_THRESH,
                max_det = self.cfg.MAX_DET,
                verbose = False,
            )
            try:    self._out.get_nowait()
            except queue.Empty: pass
            self._out.put(results[0])

    def submit(self, frame: np.ndarray):
        try:    self._in.put_nowait(frame)
        except queue.Full: pass

    def get_result(self):
        try:    return self._out.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self._running = False
        self._t.join(timeout=3)
        logging.info("[Inference] Stopped.")


# ══════════════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════════════
class Detection:
    __slots__ = ("x1","y1","x2","y2","cx","cy","label","conf","is_fruit","is_bomb")

    def __init__(self, x1, y1, x2, y2, label, conf, fruit_classes, bomb_class):
        self.x1 = x1;  self.y1 = y1
        self.x2 = x2;  self.y2 = y2
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        self.label    = label
        self.conf     = conf
        self.is_fruit = label in fruit_classes
        self.is_bomb  = label == bomb_class


def parse_detections(result, cfg: Config) -> list:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    dets  = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append(Detection(
            x1, y1, x2, y2,
            label         = names[int(box.cls[0])],
            conf          = float(box.conf[0]),
            fruit_classes = cfg.FRUIT_CLASSES,
            bomb_class    = cfg.BOMB_CLASS,
        ))
    return dets


# ══════════════════════════════════════════════════════════════════
#  SWIPE PLANNER
# ══════════════════════════════════════════════════════════════════
def _near_bomb(x, y, bombs, margin):
    for b in bombs:
        if (x - b.cx)**2 + (y - b.cy)**2 < margin**2:
            return True
    return False


def _cluster(fruits, max_dist):
    used, groups = [False] * len(fruits), []
    for i, f in enumerate(fruits):
        if used[i]: continue
        grp = [f]; used[i] = True
        for j in range(i + 1, len(fruits)):
            if used[j]: continue
            if ((fruits[j].cx-f.cx)**2+(fruits[j].cy-f.cy)**2)**0.5 <= max_dist:
                grp.append(fruits[j]); used[j] = True
        groups.append(grp)
    return groups


def plan_swipes(detections: list, cfg: Config) -> list:
    """
    Returns list of (x1, y1, x2, y2) in desktop-absolute coordinates.

    Coordinate pipeline:
        YOLO outputs coords relative to capture region
            + CLIENT_ORIGIN_X/Y
            = desktop-absolute coords for SetCursorPos
    """
    bombs  = [d for d in detections if d.is_bomb]
    fruits = [d for d in detections if d.is_fruit]
    safe   = [f for f in fruits
              if not _near_bomb(f.cx, f.cy, bombs, cfg.BOMB_SAFE_MARGIN)]
    if not safe:
        return []

    groups = _cluster(safe, cfg.GROUP_DISTANCE)
    swipes = []
    ox = cfg.CLIENT_ORIGIN_X   # 68
    oy = cfg.CLIENT_ORIGIN_Y   # 23

    for grp in groups:
        if len(grp) == 1:
            f  = grp[0]
            sx = max(0, f.x1 - cfg.SLICE_PADDING)
            ex = min(cfg.CLIENT_W, f.x2 + cfg.SLICE_PADDING)
            sy = ey = f.cy
        else:
            grp.sort(key=lambda d: d.cx)
            sx, sy = grp[0].cx, grp[0].cy
            ex, ey = grp[-1].cx, grp[-1].cy

        if (_near_bomb(sx, sy, bombs, cfg.BOMB_SAFE_MARGIN) or
                _near_bomb(ex, ey, bombs, cfg.BOMB_SAFE_MARGIN)):
            continue

        # Clamp to window bounds then convert to desktop-absolute
        sx = max(ox, min(ox + cfg.CLIENT_W, sx + ox))
        ex = max(ox, min(ox + cfg.CLIENT_W, ex + ox))
        sy = max(oy, min(oy + cfg.CLIENT_H, sy + oy))
        ey = max(oy, min(oy + cfg.CLIENT_H, ey + oy))

        swipes.append((sx, sy, ex, ey))

    return swipes


# ══════════════════════════════════════════════════════════════════
#  DEBUG OVERLAY
# ══════════════════════════════════════════════════════════════════
_CLR: dict = {}

def _color(label):
    if label not in _CLR:
        np.random.seed(hash(label) % (2**31))
        _CLR[label] = tuple(int(c) for c in np.random.randint(80, 220, 3))
    return _CLR[label]


def draw_debug(frame, dets, swipes, fps, cfg):
    vis = frame.copy()
    ox  = cfg.CLIENT_ORIGIN_X
    oy  = cfg.CLIENT_ORIGIN_Y

    for d in dets:
        if d.is_bomb:
            color = (0, 0, 230)
            text  = f"BOMB {d.conf:.2f}"
            cv2.circle(vis, (d.cx, d.cy), cfg.BOMB_SAFE_MARGIN,
                       (0, 0, 160), 1, cv2.LINE_AA)
        else:
            color = _color(d.label)
            text  = f"{d.label} {d.conf:.2f}"

        cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), color,
                      3 if d.is_bomb else 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (d.x1, d.y1-th-6), (d.x1+tw+4, d.y1), color, -1)
        cv2.putText(vis, text, (d.x1+2, d.y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(vis, (d.cx, d.cy), 4, color, -1)

    for x1, y1, x2, y2 in swipes:
        cv2.arrowedLine(vis,
                        (x1 - ox, y1 - oy),
                        (x2 - ox, y2 - oy),
                        (0, 255, 80), 2, tipLength=0.25)

    if cfg.SHOW_FPS:
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100), 2, cv2.LINE_AA)

    fn = sum(1 for d in dets if d.is_fruit)
    bn = sum(1 for d in dets if d.is_bomb)
    cv2.putText(vis, f"Fruits:{fn}  Bombs:{bn}  Swipes:{len(swipes)}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,220,50), 2, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════
#  FPS COUNTER
# ══════════════════════════════════════════════════════════════════
class FPSCounter:
    def __init__(self, n=30):
        self._t = []; self._n = n
    def tick(self):
        self._t.append(time.time())
        if len(self._t) > self._n: self._t.pop(0)
    @property
    def value(self):
        if len(self._t) < 2: return 0.0
        return (len(self._t)-1) / (self._t[-1]-self._t[0])


# ══════════════════════════════════════════════════════════════════
#  MAIN BOT
# ══════════════════════════════════════════════════════════════════
class FruitNinjaBot:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s]  %(message)s",
                            datefmt="%H:%M:%S")

        self.hwnd = _user32.FindWindowW(None, "BlueStacks App Player")
        if not self.hwnd:
            raise RuntimeError("BlueStacks App Player not found. Is it running?")
        logging.info(f"[Win32] BlueStacks hwnd: {self.hwnd}")

        model_path    = ensure_onnx(cfg)
        self.capture  = DXCamCapture(cfg.CAPTURE_REGION, target_fps=120)
        self.inferrer = InferenceThread(model_path, cfg)
        self.fps      = FPSCounter()
        self._fn      = 0
        self._dets    = []
        self._swipes  = []

    def run(self):
        logging.info("═" * 56)
        logging.info("  Bot starting in 3 seconds ...")
        logging.info("  Press  q  on the debug window to quit.")
        logging.info("═" * 56)
        time.sleep(3)

        # Focus called from main thread — required for mouse_event to work
        focus_bluestacks(self.hwnd)

        try:
            while True:
                frame = self.capture.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                self._fn += 1
                if self._fn % self.cfg.SKIP_FRAMES == 0:
                    self.inferrer.submit(frame)

                result = self.inferrer.get_result()
                if result is not None:
                    self._dets   = parse_detections(result, self.cfg)
                    self._swipes = plan_swipes(self._dets, self.cfg)

                    # Swipes execute here — in the main thread
                    # This is the only place SetCursorPos + mouse_event works
                    for x1, y1, x2, y2 in self._swipes:
                        win32_swipe(x1, y1, x2, y2,
                                    duration = self.cfg.SLICE_DURATION,
                                    steps    = self.cfg.SLICE_STEPS)
                        time.sleep(self.cfg.SWIPE_COOLDOWN)

                self.fps.tick()

                if self.cfg.SHOW_DEBUG:
                    vis = draw_debug(frame, self._dets, self._swipes,
                                     self.fps.value, self.cfg)
                    cv2.imshow("Fruit Ninja Bot  [q = quit]", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            logging.info("[Bot] Shutting down...")
            self.capture.stop()
            self.inferrer.stop()
            cv2.destroyAllWindows()
            logging.info("[Bot] Stopped.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Your model paths
    CFG.MODEL_PATH = r"C:\Fruit-Ninja\best.pt"
    CFG.ONNX_PATH  = r"C:\Fruit-Ninja\best.onnx"

    # 2. Your exact class names from data.yaml
    CFG.FRUIT_CLASSES = [
        "apple", "watermelon", "orange", "banana",
        "kiwi", "pineapple", "mango", "peach",
    ]
    CFG.BOMB_CLASS = "bomb"

    # 3. Window coords confirmed — do not change unless BlueStacks moves
    #    Client size: 664x407  |  Origin: (68, 23)

    # 4. Tune these if needed
    CFG.IMGSZ            = 320
    CFG.SKIP_FRAMES      = 2
    CFG.CONF_THRESH      = 0.50
    CFG.SLICE_DURATION   = 0.035
    CFG.SLICE_STEPS      = 12
    CFG.BOMB_SAFE_MARGIN = 55
    CFG.GROUP_DISTANCE   = 100

    bot = FruitNinjaBot(CFG)
    bot.run()