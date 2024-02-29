import numpy as np
import cv2 as cv
import time

from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

from motionDetectionBlock import MotionDetector



videos = {'web' : 0, 'camera' : '/home/yoy/Videos/YouTube/camera.mp4', 'youtube': "/home/yoy/Videos/YouTube/Advance RolePlay 7 ｜ Silver ► ВПЕРВЫЕ В GTA SAMP ► #1 [7iNp9daMfMM].mp4"}
video = videos['camera']

UseFPSLimiter = True
if video == 0:
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

delta_time = 2
timer = delta_time
bbox_list = list()


### functions ###

def change_cumsum():
    mdetector.change_cumsum_mode()
    if mdetector.cumsum_enabled:
        btn_cumsum_text.set("Disable cumsum")
    else:
        btn_cumsum_text.set("Enable cumsum")
         


### GUI ###


root = Tk()
root.title('Motion Detection')
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

app = Frame(root, bg="#748796")
app.grid()
lmain = Label(app, bg="#748796")
lmain.grid(padx=10, pady=10)

controls = Frame(root, bg="#748796")
controls.grid(column=1, row=0)

Label(controls, text="Sensitivity", bg="#748796", fg="#ecf7fc").grid(column=0, row=0, sticky=W)
sens = Scale(controls, from_=0, to=255, orient=HORIZONTAL, bg="#748796")
sens.set(10)
sens.grid(column=0, row=1, sticky=W, padx=10, pady=(10, 25))

Label(controls, text="Find bbox every Nth frame", bg="#748796", fg="#ecf7fc").grid(column=0, row=2, sticky=W)
dtime = Scale(controls, from_=1, to=42, orient=HORIZONTAL, bg="#748796")
dtime.grid(column=0, row=3, sticky=W, padx=10, pady=10)

btn_cumsum_text = StringVar()
enable_cumsum = Button(controls, bg="#748796", fg="#ecf7fc", textvariable=btn_cumsum_text, command=change_cumsum).grid(column=0, row=4, sticky=W)
btn_cumsum_text.set("Enable cumsum")


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

        mdetector.set_sens(sens.get())
        delta_time = dtime.get()

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
