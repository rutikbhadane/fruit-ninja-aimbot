import os
import time
import ctypes
import ctypes.wintypes
from PIL import ImageGrab

# ── Auto-detect the Fruit Ninja window (Google Play Games on PC) ──
_user32 = ctypes.windll.user32

def _find_fruit_ninja_hwnd():
    """
    Find the Fruit Ninja window handle.
    Handles Google Play Games title format: 'Fruit Ninja® - <username>'
    """
    found = ctypes.c_void_p(0)
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _cb(hwnd, lp):
        if not _user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetWindowTextW(hwnd, buf, 256)
        t = buf.value.strip().lower()
        if "fruit ninja" in t or "fruitninja" in t:
            found.value = hwnd
            return False  # stop enumeration
        return True

    _user32.EnumWindows(EnumProc(_cb), 0)
    return found.value


def get_window_bbox():
    """
    Returns (left, top, right, bottom) of the Fruit Ninja client area
    in screen coordinates, or None if the window is not found.
    """
    hwnd = _find_fruit_ninja_hwnd()
    if not hwnd:
        return None

    rect = ctypes.wintypes.RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = ctypes.wintypes.POINT(0, 0)
    _user32.ClientToScreen(hwnd, ctypes.byref(pt))

    left   = pt.x
    top    = pt.y
    right  = pt.x + rect.right
    bottom = pt.y + rect.bottom

    buf = ctypes.create_unicode_buffer(256)
    _user32.GetWindowTextW(hwnd, buf, 256)
    print(f"[capture] Window: '{buf.value.strip()}' — {rect.right}x{rect.bottom} at ({left}, {top})")
    return left, top, right, bottom


def capture_game_window(index=0, folder="dataset", prefix="game"):
    """
    Capture the Fruit Ninja game window and save to <folder>/<index>.png.
    The window is auto-detected — no hardcoded coordinates needed.
    """
    bbox = get_window_bbox()
    if bbox is None:
        print("[capture] ERROR: Fruit Ninja window not found. Launch the game first!")
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{index}.png"
    filepath = os.path.join(folder, filename)

    screenshot = ImageGrab.grab(bbox=bbox)
    screenshot.save(filepath)
    print(f"Screenshot saved to {filepath}")


if __name__ == "__main__":
    # Take 180 screenshots at 0.5-second intervals
    bbox = get_window_bbox()
    if bbox is None:
        print("ERROR: Launch Fruit Ninja via Google Play Games first!")
        exit(1)

    for i in range(120, 300):
        print(f"Capturing screenshot {i + 1}...")
        capture_game_window(index=i, folder="dataset/images/train")
        time.sleep(0.5)
