import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

Vid = cv.VideoCapture('cannystillframe.mp4')

if (Vid.isOpen() == False):
    print('Error vid closed')
else:
    fps = int(Vid.get(5)))
