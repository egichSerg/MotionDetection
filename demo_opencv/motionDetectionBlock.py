import numpy as np
import cv2 as cv


class MotionDetector:
    def __init__(self, resolution = (640, 480), cumsum_enabled = True, delta_time = 2) -> None:
        self.cumsum_enabled = cumsum_enabled
        self.delta_time = delta_time
        self.resolution = resolution
        self.frames_collection = np.empty(shape=resolution)


    ### set time difference in frames (for subtraction)
    def set_delta_time(self, new_delta_time):
        self.delta_time = new_delta_time

    def set_cumsum_mode(self, cumsum_mode):
        self.cumsum_enabled = cumsum_mode

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.frames_collection = np.empty(shape=resolution)

    def get_difference(self, image: np.ndarray):
        if image.shape != [ 3, *self.resolution ]:
            pass