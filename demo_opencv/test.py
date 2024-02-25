import cv2 as cv
import numpy as np
import imutils

capt = cv.VideoCapture(0)
ret, frame = capt.read()

#init backlogging frame
frame = imutils.resize(frame, 500)
frame_prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
frame_prev = cv.GaussianBlur(frame_prev, (21, 21), 0)

#min area of contours
min_area = 10000

while (True):
    
    text = 'Vse spokoino...'
    color = (0, 255, 0)
    
    ret, frame = capt.read()
    
    frame = imutils.resize(frame, 500)
    frame_copy = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    frame_copy = cv.GaussianBlur(frame_copy, (21, 21), 0)
    
    frame_delta = cv.absdiff(frame_copy, frame_prev)
    thresh = cv.threshold(frame_delta, 30, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=11)
    
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for cnt in cnts:
        if cv.contourArea(cnt) < min_area:
            continue
        (x, y, w, h) = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = 'PIDOR OBNARUZHEN!!'
        color = (0, 0, 255)
           
    cv.putText(frame, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    key = cv.waitKey(20) & 0xff    
    cv.imshow('VIDOE', frame)
    
    ret, frame = capt.read()
    
    if key == 27:
        break
    if key == ord('f'):
        frame_prev = frame_copy

cv.destroyAllWindows()
capt.release()