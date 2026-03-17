from PIL import Image
import cv2
import os
org_image = Image.open(r"fruits\fruits.png")
r_image = org_image.rotate(180)

r_image.show(r_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

path = r"dataset\train"
i = 1
while True:
    dir = os.path.dirname(path)
    img = Image.read(dir+f"img{i}")
    r_img = img.rotate(180)
    filepath = os.path.join(dir,)