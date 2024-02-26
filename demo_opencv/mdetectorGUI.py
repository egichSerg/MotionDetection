import numpy as np
import cv2 as cv
import time
import os

from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Scale, Label, StringVar
from tkinter import HORIZONTAL

from PIL import Image, ImageTk

from motionDetectionBlock import MotionDetector


### global variables ###

dir = os.path.dirname(os.path.realpath(__file__))

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(dir + r"/build/assets/frame0")

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
mdetector.set_cumsum_mode(True)
mdetector.set_sens(10)
diff = np.zeros_like(frame)

delta_time = 2
timer = delta_time
bbox_list = list()

### functions ###

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def change_cumsum():
    mdetector.change_cumsum_mode()
    if mdetector.cumsum_enabled:
        btn_cumsum_text.set("Disable cumsum")
    else:
        btn_cumsum_text.set("Enable cumsum")


### GUI ###

window = Tk()
window.title('Motion Detection')

window.geometry("950x600")
window.configure(bg = "#392D3C")


canvas = Canvas(
    window,
    bg = "#392D3C",
    height = 600,
    width = 950,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    414.0,
    375.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    361.0,
    300.0,
    image=image_image_2
)

lmain = Label(window, bg="#748796")
lmain.place(
    x=30.0,
    y=30.0,
)

canvas.create_text(
    723.0,
    140.0,
    anchor="nw",
    text="Sensitivity",
    fill="#FEFFBA",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    723.0,
    239.0,
    anchor="nw",
    text="Detect every \nNth frame",
    fill="#FEFFBA",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    723.0,
    360.0,
    anchor="nw",
    text="Cumulative sum of\nframe differences\nforget rate",
    fill="#FEFFBA",
    font=("Inter", 24 * -1)
)


sens_slider = Scale(window, from_=0, to=255, tickinterval=1, orient=HORIZONTAL, bg="#748796", fg="#ecf7fc")
sens_slider.set(10)
sens_slider.place(
    x=723.0,
    y=183.0,
    width=207.0,
    height=28.0
)

dtime = Scale(window, from_=1, to=24, tickinterval=1, orient=HORIZONTAL, bg="#748796", fg="#ecf7fc")
dtime.set(2)
dtime.place(
    x=723.0,
    y=308.0,
    width=207.0,
    height=28.0
)

cumsum_slider = Scale(window, from_=0, to=100, tickinterval=1, orient=HORIZONTAL, bg="#748796", fg="#ecf7fc")
cumsum_slider.set(20)
cumsum_slider.place(
    x=723.0,
    y=435.0,
    width=207.0,
    height=28.0
)

btn_cumsum_text = StringVar()
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
btn_cumsum = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    text="Enable cumsum",
    #textvariable=btn_cumsum_text,
    command=change_cumsum,
    relief="flat",
    fg="#FFFFFF"
)
btn_cumsum.place(
    x=723.0,
    y=500.0,
    width=207.0,
    height=51.0
)
btn_cumsum_text.set("Enable cumsum")

window.resizable(False, False)

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
        diff = cv.resize(diff, (640, 480), interpolation= cv.INTER_LINEAR)

        # showing in widget
        img = Image.fromarray(diff)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        # limit by fps and restart func
        timer += 1

        if UseFPSLimiter:
            time.sleep(max(0, time_for_frame - time.time() + start_time))

        mdetector.set_sens(sens_slider.get())
        mdetector.set_forget_percentage(cumsum_slider.get())
        delta_time = dtime.get()

        lmain.after(1, video_stream)


video_stream()

window.mainloop()
