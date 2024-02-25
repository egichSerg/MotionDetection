import numpy as np
import cv2 as cv

from motionDetectionBlock import MotionDetector

videos = {'web' : 0, 'camera' : '/home/yoy/Videos/YouTube/camera.mp4', 'youtube': "/home/yoy/Videos/YouTube/Advance RolePlay 7 ｜ Silver ► ВПЕРВЫЕ В GTA SAMP ► #1 [7iNp9daMfMM].mp4"}

capture = cv.VideoCapture(videos['youtube'])
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
        diff = mdetector.get_bbox(frame)
        timer = 0

    cv.imshow('clip', diff)
    ret, frame = capture.read() 

    timer += 1 