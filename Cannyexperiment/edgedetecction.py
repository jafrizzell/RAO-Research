from __future__ import print_function
import cv2 as cv

#backSub = cv.createBackgroundSubtractorMOG2()
backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture('canny test2_trim.mp4')

if not capture.isOpened():
    print('Unable to open: ')
    exit(0)

# Obtain frame size information using get() method
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_size = (frame_width,frame_height)
fps = int(capture.get(5))


# Initialize video writer object
output = cv.VideoWriter('Resources/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)


    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    ret, thresh = cv.threshold(fgMask, 150, 255, cv.THRESH_BINARY)
    # detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
    contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # draw contours on the original image for `CHAIN_APPROX_SIMPLE`
    image_copy1 = frame.copy()
    cv.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, cv.LINE_AA)
    # see the results
    cv.imshow('Simple approximation', image_copy1)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

'''# read image
img = cv.imread('cannystillframe_Moment.jpg')

# convert the image to grayscale format
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
# visualize the binary image
cv.imshow('Binary image', thresh)
cv.waitKey(0)
cv.imwrite('image_thres1.jpg', thresh)
cv.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = img.copy()
cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

# see the results
cv.imshow('None approximation', image_copy)
cv.waitKey(0)
cv.imwrite('contours_none_image1.jpg', image_copy)
cv.destroyAllWindows()
"""
Now let's try with `cv2.CHAIN_APPROX_SIMPLE`
"""
# detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1 = img.copy()
cv.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, cv.LINE_AA)
# see the results
cv.imshow('Simple approximation', image_copy1)
cv.waitKey(0)
cv.imwrite('contours_simple_image1.jpg', image_copy1)
cv.destroyAllWindows()'''