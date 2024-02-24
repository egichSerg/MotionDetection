import numpy as np
import cv2 as cv

from motionDetectionBlock import MotionDetector

capture = cv.VideoCapture(0)
ret, frame = capture.read() 

while ret:
    key = cv.waitKey(20) & 0xff    
    if key == 27:
        break  

    cv.imshow('clip', frame)
    ret, frame = capture.read()