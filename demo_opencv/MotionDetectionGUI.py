import numpy as np
import cv2 as cv

from motionDetectionBlock import MotionDetector

# capture = cv.VideoCapture("/home/yoy/Videos/YouTube/Advance RolePlay 7 ｜ Silver ► ВПЕРВЫЕ В GTA SAMP ► #1 [7iNp9daMfMM].mp4")
capture = cv.VideoCapture(0)
ret, frame = capture.read() 
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
print('Video resolution: ', frame.shape)

mdetector = MotionDetector(resolution=frame.shape)
mdetector.set_cumsum_mode(False)
mdetector.set_sens(30)
diff = np.zeros_like(frame)

delta_time = 1

timer = 0
while ret:
    key = cv.waitKey(20) & 0xff    
    if key == 27:
        break 
    
    if timer == delta_time:
        diff = mdetector.get_difference(frame)
        timer = 0

    cv.imshow('clip', diff)
    ret, frame = capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

    timer += 1 