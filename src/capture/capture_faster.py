import cv2
import dxcam
from PIL import Image
camera = dxcam.create()
def capture():
    #left, top = (1920 - 640) // 2= 640, (1080 - 640) // 2=220
    #right, bottom = left + 640=1280, top + 640 = 860
    #region = (left, top, right, bottom)
    region = (250,100,1000,558)
    frame = camera.grab(region=region)  # numpy.ndarray of size (640x640x3) -> (HXWXC)
    return frame
#frame = camera.grab(region)

"""
frame = capture()
cv2.imshow("Screen",frame)
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to quit
        cv2.destroyAllWindows()
        break
#Image.fromarray(frame).show()
"""