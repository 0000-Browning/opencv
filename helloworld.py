import cv2 as cv
import numpy as np
print("OpenCV:", cv.__version__)

img = np.zeros((120,400,3), dtype=np.uint8)
cv.putText(img, "OpenCV ok", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
cv.imwrite("helloworld.png", img)
