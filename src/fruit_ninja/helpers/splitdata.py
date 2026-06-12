import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32

# ── Find Fruit Ninja window (any title variant / Google Play Games username suffix)
_FRUIT_NINJA_TITLES = [
    "Fruit Ninja®",
    "Fruit Ninja",
    "Fruit Ninja Classic",
    "Fruit Ninja Classic®",
    "FruitNinja",
]

def _find_window():
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

HWND = _find_window()
if not HWND:
    raise RuntimeError("Fruit Ninja window not found. Launch the game first!")

buf = ctypes.create_unicode_buffer(256)
user32.GetWindowTextW(HWND, buf, 256)
print(f"Found window: '{buf.value.strip()}'")

rect = ctypes.wintypes.RECT()
user32.GetClientRect(HWND, ctypes.byref(rect))

pt = ctypes.wintypes.POINT(0, 0)
user32.ClientToScreen(HWND, ctypes.byref(pt))

print(f"Client size:   {rect.right} x {rect.bottom}")
print(f"Client origin: ({pt.x}, {pt.y})")
print(f"Capture region for dxcam: ({pt.x}, {pt.y}, {pt.x + rect.right}, {pt.y + rect.bottom})")