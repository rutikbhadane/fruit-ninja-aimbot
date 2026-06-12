import cv2
import numpy as np
import dxcam
import threading
import time
import ctypes
import ctypes.wintypes
from collections import deque
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════
#  CONFIG — only change these
# ══════════════════════════════════════════════════════════════════
MODEL_PATH       = r"C:\Fruit-Ninja\best_openvino_model"
FRUIT_CLASS_ID   = 1
BOMB_CLASS_ID    = 0
INFER_SIZE       = 256
CONF_THRESH      = 0.45
SKIP_FRAMES      = 2        # infer every N frames
SHOW_DEBUG       = True

# Swipe tuning
SWIPE_STEPS      = 15       # more = smoother, slower
SWIPE_STEP_DELAY = 0.003    # seconds between steps
SWIPE_LEAD       = 60       # px before and after fruit center
BOMB_MARGIN      = 40       # px exclusion zone around bombs
BETWEEN_SWIPES   = 0.01     # seconds between consecutive swipes

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — Find Fruit Ninja window + compute regions automatically
# ══════════════════════════════════════════════════════════════════
user32 = ctypes.windll.user32

_FRUIT_NINJA_TITLES = [
    "Fruit Ninja®",
    "Fruit Ninja",
    "Fruit Ninja Classic",
    "Fruit Ninja Classic®",
    "FruitNinja",
]

def _find_fruit_ninja_hwnd():
    """Return HWND for any visible window whose title contains 'Fruit Ninja'."""
    # Fast path: exact known titles
    for title in _FRUIT_NINJA_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
    # Slow path: enumerate all visible windows
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
        t = buf.value.strip().lower()
        if "fruit ninja" in t or "fruitninja" in t:
            found.value = hwnd
            return False
        return True
    user32.EnumWindows(EnumProc(_cb), 0)
    return found.value

HWND = _find_fruit_ninja_hwnd()
if not HWND:
    raise RuntimeError("Fruit Ninja window not found. Launch the game first!")

buf = ctypes.create_unicode_buffer(256)
user32.GetWindowTextW(HWND, buf, 256)
print(f"Found window: '{buf.value.strip()}'")

# Get client area size (excludes title bar, borders)
_rect = ctypes.wintypes.RECT()
user32.GetClientRect(HWND, ctypes.byref(_rect))
CLIENT_W = _rect.right
CLIENT_H = _rect.bottom

# Get where client area starts on desktop (accounts for title bar etc.)
_pt = ctypes.wintypes.POINT(0, 0)
user32.ClientToScreen(HWND, ctypes.byref(_pt))
CLIENT_ORIGIN_X = _pt.x
CLIENT_ORIGIN_Y = _pt.y

# dxcam capture region = exact client area on desktop
CAPTURE_REGION = (
    CLIENT_ORIGIN_X,
    CLIENT_ORIGIN_Y,
    CLIENT_ORIGIN_X + CLIENT_W,
    CLIENT_ORIGIN_Y + CLIENT_H,
)

print(f"Fruit Ninja client: {CLIENT_W}x{CLIENT_H} at ({CLIENT_ORIGIN_X}, {CLIENT_ORIGIN_Y})")
print(f"Capture region: {CAPTURE_REGION}")

# ══════════════════════════════════════════════════════════════════
#  STEP 2 — Input: SetCursorPos + mouse_event (Method A)
# ══════════════════════════════════════════════════════════════════
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004
_mouse_event         = ctypes.windll.user32.mouse_event
_SetCursorPos        = ctypes.windll.user32.SetCursorPos
_SetForeground       = ctypes.windll.user32.SetForegroundWindow

def _focus_game_window():
    """Bring the Fruit Ninja window to foreground so it receives input."""
    _SetForeground(HWND)
    time.sleep(0.05)  # give Windows time to switch focus

def swipe(x1, y1, x2, y2):
    """
    Smooth swipe from (x1,y1) to (x2,y2) in desktop coordinates.
    Uses SetCursorPos + mouse_event.
    """
    _SetCursorPos(int(x1), int(y1))
    time.sleep(0.008)
    _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.005)

    for i in range(1, SWIPE_STEPS + 1):
        t  = i / SWIPE_STEPS
        ix = int(x1 + (x2 - x1) * t)
        iy = int(y1 + (y2 - y1) * t)
        _SetCursorPos(ix, iy)
        time.sleep(SWIPE_STEP_DELAY)

    _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def slice_fruit(rel_cx, rel_cy):
    """
    Horizontal swipe through a fruit at (rel_cx, rel_cy)
    which are coordinates relative to the capture region.
    Converts to desktop coords before swiping.
    """
    # clamp to client bounds before converting — can never go off screen
    rel_cx = max(SWIPE_LEAD, min(CLIENT_W - SWIPE_LEAD, rel_cx))
    rel_cy = max(5, min(CLIENT_H - 5, rel_cy))

    # convert: capture-relative → desktop absolute
    desk_x  = CLIENT_ORIGIN_X + rel_cx
    desk_y  = CLIENT_ORIGIN_Y + rel_cy
    start_x = desk_x - SWIPE_LEAD
    end_x   = desk_x + SWIPE_LEAD

    swipe(start_x, desk_y, end_x, desk_y)

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — Result parsing (vectorised, no per-box Python loop)
# ══════════════════════════════════════════════════════════════════
def parse(result):
    """
    Returns:
        fruit_centers : list of (cx, cy) relative to capture region
        bomb_boxes    : list of (x1, y1, x2, y2) relative to capture region
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    coords  = boxes.xyxy.cpu().numpy()

    bomb_coords  = coords[cls_ids == BOMB_CLASS_ID]
    fruit_coords = coords[cls_ids == FRUIT_CLASS_ID]

    bomb_boxes = bomb_coords.tolist()

    if len(fruit_coords) == 0:
        return [], bomb_boxes

    cx = (fruit_coords[:, 0] + fruit_coords[:, 2]) / 2
    cy = (fruit_coords[:, 1] + fruit_coords[:, 3]) / 2
    return list(zip(cx.tolist(), cy.tolist())), bomb_boxes

def is_safe(fx, fy, bomb_boxes):
    for bx1, by1, bx2, by2 in bomb_boxes:
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2
        if (fx - bcx)**2 + (fy - bcy)**2 < BOMB_MARGIN**2:
            return False
    return True

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — Nearest-neighbour sort (minimise cursor travel)
# ══════════════════════════════════════════════════════════════════
def sort_nearest(centers):
    if len(centers) <= 1:
        return centers
    remaining = list(centers)
    path = [remaining.pop(0)]
    while remaining:
        last = path[-1]
        idx  = min(range(len(remaining)),
                   key=lambda i: (remaining[i][0]-last[0])**2 +
                                 (remaining[i][1]-last[1])**2)
        path.append(remaining.pop(idx))
    return path

# ══════════════════════════════════════════════════════════════════
#  STEP 5 — Threaded bot
# ══════════════════════════════════════════════════════════════════
class FruitNinjaBot:
    def __init__(self):
        print("Loading model...")
        self.model = YOLO(MODEL_PATH, task='detect')

        # Warmup — first inference is always slow
        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(dummy, verbose=False,
                               imgsz=INFER_SIZE, conf=CONF_THRESH)
        print("Model warmed up.")

        # dxcam in continuous mode
        self.cam = dxcam.create(output_color="BGR")
        self.cam.start(region=CAPTURE_REGION, target_fps=60, video_mode=True)

        self._frame_buf  = deque(maxlen=1)  # always latest frame
        self._result_buf = deque(maxlen=1)  # always latest result
        self._stop       = threading.Event()

    # ── Thread: capture ───────────────────────────────────────────
    def _capture_loop(self):
        get = self.cam.get_latest_frame
        buf = self._frame_buf
        while not self._stop.is_set():
            f = get()
            if f is not None:
                buf.append(f)

    # ── Thread: inference ─────────────────────────────────────────
    def _infer_loop(self):
        predict  = self.model.predict
        fbuf     = self._frame_buf
        rbuf     = self._result_buf
        n        = 0
        while not self._stop.is_set():
            if not fbuf:
                continue
            n += 1
            if n % SKIP_FRAMES != 0:
                continue
            frame   = fbuf[-1]
            results = predict(frame, verbose=False,
                              imgsz=INFER_SIZE, conf=CONF_THRESH)
            rbuf.append((frame, results[0]))

    # ── Main loop: slice + draw ───────────────────────────────────
    def run(self):
        print("\n" + "="*50)
        print("Bot starting — focusing Fruit Ninja in 2s...")
        print("="*50)
        time.sleep(2)
        _focus_game_window()

        t_cap = threading.Thread(target=self._capture_loop, daemon=True)
        t_inf = threading.Thread(target=self._infer_loop,   daemon=True)
        t_cap.start()
        t_inf.start()

        if SHOW_DEBUG:
            cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug", 640, 400)

        frame_count = 0
        fps_val     = 0.0
        t_fps       = time.perf_counter()

        while not self._stop.is_set():
            if not self._result_buf:
                continue

            frame, result        = self._result_buf[-1]
            fruit_centers, bombs = parse(result)
            safe = [fc for fc in fruit_centers if is_safe(fc[0], fc[1], bombs)]

            if safe:
                # Re-focus game window every 60 frames
                # (OpenCV debug window can steal focus)
                if frame_count % 60 == 0:
                    _focus_game_window()

                ordered = sort_nearest(safe)
                for fx, fy in ordered:
                    slice_fruit(fx, fy)
                    time.sleep(BETWEEN_SWIPES)

            # ── Debug overlay ─────────────────────────────────────
            if SHOW_DEBUG:
                disp = frame.copy()

                for bx1, by1, bx2, by2 in bombs:
                    cv2.rectangle(disp,
                                  (int(bx1), int(by1)),
                                  (int(bx2), int(by2)),
                                  (0, 0, 255), 2)
                    bcx = int((bx1+bx2)/2)
                    bcy = int((by1+by2)/2)
                    cv2.circle(disp, (bcx, bcy),
                               BOMB_MARGIN, (0, 0, 180), 1)

                for fx, fy in fruit_centers:
                    color = (0, 255, 0) if is_safe(fx, fy, bombs) \
                            else (0, 255, 255)
                    cv2.circle(disp, (int(fx), int(fy)), 6, color, -1)
                    # Show swipe line
                    cv2.line(disp,
                             (int(fx)-SWIPE_LEAD, int(fy)),
                             (int(fx)+SWIPE_LEAD, int(fy)),
                             (255, 100, 0), 1)

                frame_count += 1
                now = time.perf_counter()
                if now - t_fps >= 1.0:
                    fps_val     = frame_count / (now - t_fps)
                    frame_count = 0
                    t_fps       = now

                cv2.putText(disp, f"FPS:{fps_val:.1f}  Fruits:{len(fruit_centers)}  Bombs:{len(bombs)}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)
                cv2.imshow("Debug", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._stop.set()
        self.cam.stop()
        cv2.destroyAllWindows()
        print("Bot stopped.")

if __name__ == "__main__":
    bot = FruitNinjaBot()
    bot.run()