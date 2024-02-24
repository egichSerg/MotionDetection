import numpy as np
import cv2 as cv

class MotionDetector:
    def __init__(self, resolution = (640, 480), forget_percentage = 50, cumsum_enabled = True) -> None:
        self.cumsum_enabled = cumsum_enabled
        self.resolution = resolution
        self.forget_percentage = forget_percentage
        self.frame_difference = np.empty(shape=resolution)
        self.prev_frame = None
        self.frame_cumsum = np.zeros(shape=resolution)


    ### set time difference in frames (for subtraction)
    def set_forget_percentage(self, new_forget_percentage):
        self.forget_percentage = new_forget_percentage

    def set_cumsum_mode(self, cumsum_mode):
        self.cumsum_enabled = cumsum_mode

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.frame_difference = np.empty(shape=resolution)
        self.prev_frame = np.empty(shape=resolution)
        self.frame_cumsum = np.zeros(shape=resolution)

    def get_difference(self, frame: np.ndarray):
        if frame.shape != self.resolution:
            raise SystemExit(f"Resolution of detector {self.resolution} and image {frame.shape} doesn't match")
        
        if self.prev_frame is None:
            self.prev_frame = frame
        
        self.frame_difference = frame - self.prev_frame
        _, self.frame_difference = cv.threshold(self.frame_difference, 100, 255, cv.THRESH_TOZERO)
        output = self.frame_difference

        if self.cumsum_enabled:
            self.frame_cumsum = self.frame_cumsum * (self.forget_percentage / 100) + self.frame_difference
            output = self.frame_cumsum

        self.prev_frame = frame

        return output