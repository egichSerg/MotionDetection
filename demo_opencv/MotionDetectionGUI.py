import numpy as np
import cv2 as cv
import time

from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

from motionDetectionBlock import MotionDetector

videos = {'web' : 0, 'camera' : '/home/yoy/Videos/YouTube/camera.mp4', 'youtube': "/home/yoy/Videos/YouTube/Advance RolePlay 7 ｜ Silver ► ВПЕРВЫЕ В GTA SAMP ► #1 [7iNp9daMfMM].mp4"}
video = videos['youtube']

UseFPSLimiter = True
if video == videos['web']:
      UseFPSLimiter = False


capture = cv.VideoCapture(video)
ret, frame = capture.read() 
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
print('Video resolution: ', frame.shape) # should loop

fps = capture.get(cv.CAP_PROP_FPS)
time_for_frame = 1/fps

mdetector = MotionDetector(resolution=frame.shape)
mdetector.set_cumsum_mode(False)
mdetector.set_sens(10)
diff = np.zeros_like(frame)

delta_time = 1
timer = delta_time
bbox_list = list()


### GUI ###


root = Tk()
root.title('Motion Detection')

app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

def video_stream():
        
        global timer
        global delta_time
        global bbox_list

        # read frame
        start_time = time.time()
        ret, frame = capture.read() 

        if timer == delta_time:
            bbox_list = mdetector.get_bbox(frame)
            timer = 0
        
        diff = mdetector.draw_bbox(frame, bbox_list)

        # showing in widget
        img = Image.fromarray(diff)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        # limit by fps and restart func
        timer += 1

        if UseFPSLimiter:
            time.sleep(max(0, time_for_frame - time.time() + start_time))

        lmain.after(1, video_stream)


video_stream()
root.mainloop()


# while ret:
#     key = cv.waitKey(20) & 0xff    
#     if key == 27:
#         break 
    
#     if timer == delta_time:
#         diff = mdetector.get_bbox(frame)
#         timer = 0

#     cv.imshow('clip', diff)
#     ret, frame = capture.read() 

#     timer += 1 