import cv2
import numpy as np

cap = cv2.VideoCapture('cannytest3_Trim.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
#print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    _, thresh = cv2.threshold(blur,255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(thresh,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)
    dilate = cv2.dilate(thresh, None, iterations=11)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 100 or 400 < area:
            continue
        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame1, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape


    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    cv2.imshow('thresh',dilate)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()