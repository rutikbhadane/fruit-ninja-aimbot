import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

# ── Find Fruit Ninja window (auto-detected) ───────────────────────
_FRUIT_NINJA_TITLES = [
    "Fruit Ninja®",
    "Fruit Ninja",
    "Fruit Ninja Classic",
    "Fruit Ninja Classic®",
    "FruitNinja",
]

def _find_window():
    for title in _FRUIT_NINJA_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
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

HWND_BS = _find_window()
if not HWND_BS:
    raise RuntimeError("Fruit Ninja window not found. Launch the game first!")

_title_buf = ctypes.create_unicode_buffer(256)
user32.GetWindowTextW(HWND_BS, _title_buf, 256)
print(f"Found window: '{_title_buf.value.strip()}'")

# ── Get client info ──────────────────────────────────────────────
rect = ctypes.wintypes.RECT()
user32.GetClientRect(HWND_BS, ctypes.byref(rect))
print(f"Client size: {rect.right} x {rect.bottom}")

pt = ctypes.wintypes.POINT(0, 0)
user32.ClientToScreen(HWND_BS, ctypes.byref(pt))
print(f"Client origin on desktop: ({pt.x}, {pt.y})")

cx = rect.right  // 2
cy = rect.bottom // 2
# Absolute desktop coords of game center
sx = pt.x + cx
sy = pt.y + cy

print(f"Game center on desktop: ({sx}, {sy})")

# ── Helpers ──────────────────────────────────────────────────────
MOUSEEVENTF_MOVE      = 0x0001
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_ABSOLUTE  = 0x8000

SW  = ctypes.windll.user32.GetSystemMetrics(0)
SH  = ctypes.windll.user32.GetSystemMetrics(1)

def abs_coords(x, y):
    return int(x * 65535 / SW), int(y * 65535 / SH)

def method_A_swipe(start_x, start_y, end_x, end_y):
    """SetCursorPos + mouse_event — physically moves real cursor"""
    user32.SetForegroundWindow(HWND_BS)
    time.sleep(0.1)
    user32.SetCursorPos(start_x, start_y)
    time.sleep(0.02)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    steps = 15
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(start_x + (end_x - start_x) * t)
        iy = int(start_y + (end_y - start_y) * t)
        user32.SetCursorPos(ix, iy)
        time.sleep(0.004)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def method_B_swipe(start_x, start_y, end_x, end_y):
    """SendInput absolute — the method in current bot"""
    def move_click(x, y, flags):
        ax, ay = abs_coords(x, y)
        class MI(ctypes.Structure):
            _fields_ = [("dx",ctypes.c_long),("dy",ctypes.c_long),
                        ("mouseData",ctypes.c_ulong),("dwFlags",ctypes.c_ulong),
                        ("time",ctypes.c_ulong),("dwExtraInfo",ctypes.c_void_p)]
        class II(ctypes.Union):
            _fields_ = [("mi", MI)]
        class INP(ctypes.Structure):
            _fields_ = [("type",ctypes.c_ulong),("ii",II)]
        inp = INP(0, II(mi=MI(ax, ay, 0, flags, 0, None)))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    user32.SetForegroundWindow(HWND_BS)
    time.sleep(0.1)
    move_click(start_x, start_y,
               MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN)
    steps = 15
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(start_x + (end_x - start_x) * t)
        iy = int(start_y + (end_y - start_y) * t)
        move_click(ix, iy, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        time.sleep(0.004)
    move_click(end_x, end_y,
               MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP)

def method_C_swipe(start_x, start_y, end_x, end_y):
    """PostMessageW directly to hwnd — no focus needed"""
    WM_MOUSEMOVE   = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP   = 0x0202
    def LP(x, y):  # coords relative to client area
        cx_ = x - pt.x
        cy_ = y - pt.y
        return (cy_ << 16) | (cx_ & 0xFFFF)
    user32.PostMessageW(HWND_BS, WM_LBUTTONDOWN, 1, LP(start_x, start_y))
    time.sleep(0.01)
    steps = 15
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(start_x + (end_x - start_x) * t)
        iy = int(start_y + (end_y - start_y) * t)
        user32.PostMessageW(HWND_BS, WM_MOUSEMOVE, 1, LP(ix, iy))
        time.sleep(0.004)
    user32.PostMessageW(HWND_BS, WM_LBUTTONUP, 0, LP(end_x, end_y))

# ── Run all 3 tests ──────────────────────────────────────────────
# Swipe left→right across game center, 150px wide
X1 = sx - 75
X2 = sx + 75
Y  = sy

print("\n--- TEST A: SetCursorPos + mouse_event ---")
print("Watch the game for a slash. Starting in 3s...")
time.sleep(3)
method_A_swipe(X1, Y, X2, Y)
print("Done. Slash? (remember A)")
time.sleep(2)

print("\n--- TEST B: SendInput absolute ---")
print("Starting in 3s...")
time.sleep(3)
method_B_swipe(X1, Y, X2, Y)
print("Done. Slash? (remember B)")
time.sleep(2)

print("\n--- TEST C: PostMessageW to hwnd ---")
print("Starting in 3s...")
time.sleep(3)
method_C_swipe(X1, Y, X2, Y)
print("Done. Slash? (remember C)")

print("\nWhich methods showed a slash in the game?")