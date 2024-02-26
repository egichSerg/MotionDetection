import os

from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Scale
from tkinter import HORIZONTAL

dir = os.path.dirname(os.path.realpath(__file__))

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(dir + r"/build/assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


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

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    350.0,
    270.0,
    image=image_image_3
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

# entry_image_1 = PhotoImage(
#     file=relative_to_assets("entry_1.png"))
# entry_bg_1 = canvas.create_image(
#     826.5,
#     198.0,
#     image=entry_image_1
# )
# entry_1 = Text(
#     bd=0,
#     bg="#D9D9D9",
#     fg="#000716",
#     highlightthickness=0
# )
# entry_1.place(
#     x=723.0,
#     y=183.0,
#     width=207.0,
#     height=28.0
# )

sens_scale = Scale(window, from_=0, to=255, tickinterval=1, orient=HORIZONTAL, bg="#748796", fg="#ecf7fc")
sens_scale.place(
    x=723.0,
    y=183.0,
    width=207.0,
    height=28.0
)

dtime = Scale(window, from_=0, to=24, tickinterval=1, orient=HORIZONTAL, bg="#748796", fg="#ecf7fc")
dtime.place(
    x=723.0,
    y=308.0,
    width=207.0,
    height=28.0
)

# entry_image_2 = PhotoImage(
#     file=relative_to_assets("entry_2.png"))
# entry_bg_2 = canvas.create_image(
#     826.5,
#     323.0,
#     image=entry_image_2
# )
# entry_2 = Text(
#     bd=0,
#     bg="#D9D9D9",
#     fg="#000716",
#     highlightthickness=0
# )
# entry_2.place(
#     x=723.0,
#     y=308.0,
#     width=207.0,
#     height=28.0
# )

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=723.0,
    y=408.0,
    width=207.0,
    height=51.0
)
window.resizable(False, False)
window.mainloop()
