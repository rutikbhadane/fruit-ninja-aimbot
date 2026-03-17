import cv2
import dxcam
import numpy as np
import time
import keyboard
import ctypes
from ultralytics import YOLO

# --- Low-level SendInput structures for reliable game input ---
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# Cache screen resolution once at startup (avoid syscall per mouse move)
SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUT_UNION),
    ]

# Pre-allocate a single INPUT struct to reuse (avoid allocation per call)
_cached_input = INPUT()
_cached_input.type = INPUT_MOUSE
_cached_input.union.mi.mouseData = 0
_cached_input.union.mi.time = 0
_cached_input.union.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
_send_input = ctypes.windll.user32.SendInput
_sizeof_input = ctypes.sizeof(_cached_input)

def send_mouse(flags, x=0, y=0):
    """Ultra-fast mouse event using a pre-allocated SendInput struct."""
    _cached_input.union.mi.dx = int(x * 65535 / SCREEN_W)
    _cached_input.union.mi.dy = int(y * 65535 / SCREEN_H)
    _cached_input.union.mi.dwFlags = flags | MOUSEEVENTF_ABSOLUTE
    _send_input(1, ctypes.byref(_cached_input), _sizeof_input)

class FruitNinjaBot:
    def __init__(self, model_path, region=(250, 100, 1000, 558), debug=True, infer_size=320):
        print("Initializing bot...")
        self.camera = dxcam.create()
        self.region = region
        self.debug = debug
        self.infer_size = infer_size
        
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Classes: 1 is Fruit, 0 is Bomb
        self.CLASS_FRUIT = 1
        self.CLASS_BOMB = 0
        
        self.is_running = False
        self.show_vision = True
        
        # Pre-compute region offsets
        self.offset_x = region[0]
        self.offset_y = region[1]

    def get_frame(self):
        return self.camera.grab(region=self.region)

    def slice_through_fruits(self, fruit_centers):
        """
        Ultra-fast continuous LEFT-CLICK DRAG through all fruit centers.
        Small delays between moves are required for the game to register the drag.
        """
        if self.debug or not fruit_centers:
            return
        
        # Sort left-to-right
        fruit_centers.sort(key=lambda p: p[0])
        
        first_x, first_y = int(fruit_centers[0][0]), int(fruit_centers[0][1])
        
        # Move + press down
        send_mouse(MOUSEEVENTF_MOVE, first_x, first_y)
        time.sleep(0.001)
        send_mouse(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_MOVE, first_x, first_y)
        time.sleep(0.001)
        
        # Drag through each fruit with 3 interpolation steps
        prev_x, prev_y = first_x, first_y
        steps = 3
        for center in fruit_centers:
            tx, ty = int(center[0]), int(center[1])
            dx = tx - prev_x
            dy = ty - prev_y
            for i in range(1, steps + 1):
                send_mouse(MOUSEEVENTF_MOVE, 
                          prev_x + dx * i // steps, 
                          prev_y + dy * i // steps)
                time.sleep(0.001)  # 1ms per step - minimum for game to see the drag
            prev_x, prev_y = tx, ty
        
        # Overshoot + release
        send_mouse(MOUSEEVENTF_MOVE, prev_x + 40, prev_y)
        time.sleep(0.001)
        send_mouse(MOUSEEVENTF_LEFTUP | MOUSEEVENTF_MOVE, prev_x + 40, prev_y)

    def get_fruit_screen_center(self, box):
        x_left, y_top, x_right, y_bottom = box
        return (
            (x_left + x_right) / 2 + self.offset_x,
            ((y_top + y_bottom) / 2) + 15 + self.offset_y  # +15 gravity compensation
        )

    def run(self):
        print("\n" + "="*40)
        if self.debug:
            print("BOT STARTED IN DEBUG MODE.")
        else:
            print("BOT STARTED IN LIVE MODE.")
            print("WARNING: THE BOT WILL TAKE CONTROL OF YOUR MOUSE.")
        print(f"YOLO inference size: {self.infer_size}x{self.infer_size}")
        print("PRESS 'q' OR 'esc' TO STOP.")
        print("="*40 + "\n")

        self.is_running = True
        
        if self.show_vision:
            cv2.namedWindow("Fruit Ninja Vision")
            cv2.moveWindow("Fruit Ninja Vision", 1100, 100)
        
        frame_count = 0
        loop_start = time.perf_counter()
        
        while self.is_running:
            t0 = time.perf_counter()
            
            # Check kill switch every 5 frames instead of every frame
            if frame_count % 5 == 0:
                if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                    print("Emergency stop detected.")
                    break
            
            # --- Screen Capture ---
            frame = self.get_frame()
            if frame is None:
                continue
            t_capture = time.perf_counter()
                
            # --- YOLO Inference ---
            results = self.model.predict(
                source=frame, 
                verbose=False, 
                conf=0.45,
                imgsz=self.infer_size
            )
            t_infer = time.perf_counter()
            
            # --- Parse Results ---
            result = results[0]
            fruit_centers = []
            bombs_present = False
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                coords = box.xyxy[0].tolist()
                
                if cls_id == self.CLASS_BOMB:
                    bombs_present = True
                elif cls_id == self.CLASS_FRUIT:
                    fruit_centers.append(self.get_fruit_screen_center(coords))

            # --- Slice FIRST, draw AFTER (action before visuals!) ---
            if not bombs_present and fruit_centers:
                self.slice_through_fruits(fruit_centers)
            t_slice = time.perf_counter()
            
            # --- Vision window (draw AFTER slicing so it doesn't delay action) ---
            if self.show_vision:
                display_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    
                    if cls_id == self.CLASS_BOMB:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, "BOMB", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif cls_id == self.CLASS_FRUIT:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                if bombs_present:
                    cv2.putText(display_frame, "BOMB - HOLD", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show FPS and timing breakdown
                t_total = time.perf_counter() - t0
                fps_text = f"Capture:{(t_capture-t0)*1000:.0f}ms | Infer:{(t_infer-t_capture)*1000:.0f}ms | Slice:{(t_slice-t_infer)*1000:.0f}ms | Total:{t_total*1000:.0f}ms"
                cv2.putText(display_frame, fps_text, (5, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow("Fruit Ninja Vision", display_frame)
                cv2.waitKey(1)
            
            frame_count += 1

        if self.show_vision:
            cv2.destroyAllWindows()
        
        elapsed = time.perf_counter() - loop_start
        if frame_count > 0:
            print(f"Ran {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
        print("Bot terminated safely.")

if __name__ == "__main__":
    YOLO_MODEL_PATH = "best.pt" 
    GAME_REGION = (250, 100, 1000, 558) 
    DEBUG_MODE = False
    
    # Inference size (multiple of 32). Lower = faster. Try 256 for max speed.
    INFER_SIZE = 320
    
    bot = FruitNinjaBot(
        model_path=YOLO_MODEL_PATH, 
        region=GAME_REGION, 
        debug=DEBUG_MODE,
        infer_size=INFER_SIZE
    )
    bot.run()
