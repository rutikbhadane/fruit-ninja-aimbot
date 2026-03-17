import time
import dxcam
from fast_ctypes_screenshots import (
    ScreenshotOfWindow,
    ScreenshotOfOneMonitor
)

def benchmark_dxcam(region, iterations=100):
    camera = dxcam.create(output_color="BGR")
    # Warmup
    for _ in range(10):
        camera.grab(region=region)
        
    start_time = time.time()
    for _ in range(iterations):
        frame = camera.grab(region=region)
        while frame is None:
            frame = camera.grab(region=region)
    end_time = time.time()
    
    fps = iterations / (end_time - start_time)
    print(f"DXCam Average FPS: {fps:.2f} ({iterations} frames in {end_time - start_time:.4f}s)")
    camera.stop()
    return fps

def benchmark_ctypes(region, iterations=100):
    # We will grab the whole monitor and slice it, since ctypes doesn't take regions directly usually
    # or use Window if we have the hwnd. For pure monitor speed:
    # fast_ctypes takes: monitor (int default 0)
    
    # Warmup
    for _ in range(10):
        with ScreenshotOfOneMonitor(
            monitor=0,
            ascontiguousarray=True
        ) as screenshots_monitor:
            img = screenshots_monitor.screenshot_one_monitor()
            
    start_time = time.time()
    for _ in range(iterations):
        with ScreenshotOfOneMonitor(
            monitor=0,
            ascontiguousarray=False # faster
        ) as screenshots_monitor:
            img = screenshots_monitor.screenshot_one_monitor()
            # Crop to region (left, top, right, bottom)
            # img is BGR numpy array
            crop = img[region[1]:region[3], region[0]:region[2]]
            
    end_time = time.time()
    
    fps = iterations / (end_time - start_time)
    print(f"fast_ctypes_screenshots Average FPS: {fps:.2f} ({iterations} frames in {end_time - start_time:.4f}s)")
    return fps

if __name__ == "__main__":
    region = (250, 100, 1000, 558)
    print(f"Benchmarking region: {region}")
    
    d_fps = benchmark_dxcam(region)
    benchmark_ctypes(region)
