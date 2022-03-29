import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

Vid = cv.VideoCapture('cannystillframe.mp4')

if (Vid.isOpen() == False):
    print('Error vid closed')
else:
    fps = int(Vid.get(5))
    print("Frame Rate : ",fps,"frames per second")
    frame_count = Vid.get(7)
    print("Frame count : ", frame_count)

while(Vid.isOpened()):
    # vCapture.read() methods returns a tuple, first element is a bool
    # and the second is frame

    ret, frame = Vid.read()
    if ret == True:
        cv.imshow('Frame',frame)
        k = cv.waitKey(20)
        # 113 is ASCII code for q key
        if k == 113:
            break
    else:
        break

# Obtain frame size information using get() method
frame_width = int(Vid.get(3))
frame_height = int(Vid.get(4))
frame_size = (frame_width,frame_height)
fps = fps