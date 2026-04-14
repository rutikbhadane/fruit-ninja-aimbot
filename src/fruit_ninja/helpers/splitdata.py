import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32
HWND   = user32.FindWindowW(None, "BlueStacks App Player")

rect = ctypes.wintypes.RECT()
user32.GetClientRect(HWND, ctypes.byref(rect))

pt = ctypes.wintypes.POINT(0, 0)
user32.ClientToScreen(HWND, ctypes.byref(pt))

print(f"Client size:   {rect.right} x {rect.bottom}")
print(f"Client origin: ({pt.x}, {pt.y})")
print(f"Capture region for dxcam: ({pt.x}, {pt.y}, {pt.x + rect.right}, {pt.y + rect.bottom})")