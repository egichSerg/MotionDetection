import numpy as np
import cv2 as cv

from motionDetectionBlock import MotionDetector

capture = cv.VideoCapture("/home/yoy/Videos/YouTube/Advance RolePlay 7 ｜ Silver ► ВПЕРВЫЕ В GTA SAMP ► #1 [7iNp9daMfMM].mp4")
ret, frame = capture.read() 
print(frame.shape)

while ret:
    key = cv.waitKey(20) & 0xff    
    if key == 27:
        break  

    cv.imshow('clip', frame)
    ret, frame = capture.read()