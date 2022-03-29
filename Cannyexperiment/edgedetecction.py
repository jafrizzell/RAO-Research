import numpy as np
import cv2
from skimage import data, filters
from matplotlib import pyplot as plt

# Open Video
cap = cv2.VideoCapture('canny test1.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

''''# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)'''

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):

    # Read frame
    ret, frame = cap.read()
    # Convert current frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    # Display image
    cv2.imshow('frame', dframe)
    cv2.waitKey(20)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

'''if (Vid.isOpened() == False):
    print('Error vid closed')
else:
    fps = int(Vid.get(5))
    print("Frame Rate : ",fps,"frames per second")
    frame_count = Vid.get(7)
    print("Frame count : ", frame_count)
# Obtain frame size information using get() method
frame_width = int(Vid.get(3))
frame_height = int(Vid.get(4))
frame_size = (frame_width,frame_height)

while(Vid.isOpened()):
    # vCapture.read() methods returns a tuple, first element is a bool
    # and the second is frame

    ret, frame = Vid.read()
    if ret == True:
        cv.imshow('Frame',frame)
        k = cv.waitKey(fps)
        # 113 is ASCII code for q key
        if k == 113:
            break
    else:
        break
# Initialize video writer object
output = cv.VideoWriter('Resources/output_video_from_file.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

while(Vid.isOpened()):
    ret, frame = Vid.read()
    if ret == True:
        # Write the frame to the output files
        output.write(frame)
    else:
        print('Stream disconnected')
        break

# Release the objects
Vid.release()
output.release()'''