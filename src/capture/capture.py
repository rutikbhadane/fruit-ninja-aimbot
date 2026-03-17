from capture_faster import capture
from PIL import Image
import os
import cv2
import numpy as np
import time
while True:
    #directory = os.path.dirname("dataset")
    time.sleep(2)
    for i in range(415,1000):
        directory = os.path.dirname("dataset")
        pic = f"img{i}.png"
        frame = capture()
        img = cv2.cvtColor(np.array(frame),cv2.COLOR_RGB2BGR)
        #image = np.array(frame)
        filepath = os.path.join(directory,pic)
        cv2.imwrite(filepath,img)
        print("image saved to "+f"dataset/img{i}.")
        time.sleep(0.5)
        
    