import os
import time
from PIL import ImageGrab

def capture_game_window(x=100, y=100, width=640, height=384, folder="dataset", prefix="game"):
    # Ensure dataset folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define bounding box (left, top, right, bottom)
    bbox = (x, y, x + width, y + height)

    # Timestamp for unique filename
    timestamp = int(time.time())
    filename = f"{i}.png"
    filepath = os.path.join(folder, filename)

    # Capture screenshot
    screenshot = ImageGrab.grab(bbox=bbox)
    screenshot.save(filepath)

    print(f"Screenshot saved to {filepath}")

if __name__ == "__main__":
    # Example: take 5 screenshots at 3-second intervals
    for i in range(120,300):
        print(f"Capturing screenshot {i+1}...")
        capture_game_window(100, 100, 640, 384, folder="dataset/images/train")
        time.sleep(0.5)
