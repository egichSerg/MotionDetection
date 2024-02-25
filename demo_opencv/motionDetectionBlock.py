import numpy as np
import cv2 as cv
import imutils

class MotionDetector:
    def __init__(self, resolution = (640, 480), forget_percentage = 50, min_area_detect = 10000, cumsum_enabled = True) -> None:
        self.cumsum_enabled = cumsum_enabled
        self.resolution = resolution
        self.forget_percentage = forget_percentage
        self.frame_difference = np.empty(shape=resolution)
        self.prev_frame = None
        self.frame_cumsum = np.zeros(shape=resolution)
        self.sens = 30
        self.min_area = 10000


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

    def set_sens(self, new_sens):
        self.sens = new_sens

    def get_difference(self, frame: np.ndarray):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if frame.shape != self.resolution:
            raise SystemExit(f"Resolution of detector {self.resolution} and image {frame.shape} doesn't match")
        
        if self.prev_frame is None:
            self.prev_frame = frame
        
        self.frame_difference = cv.absdiff(frame, self.prev_frame)
        output = self.frame_difference

        if self.cumsum_enabled:
            self.frame_cumsum = self.frame_cumsum * (self.forget_percentage / 100) + self.frame_difference
            output = self.frame_cumsum

        self.prev_frame = frame

        return output
    
    def get_bbox(self, frame: np.ndarray):
        diff = self.get_difference(frame)

        diff = cv.threshold(diff, self.sens, 255, cv.THRESH_BINARY)[1]
        diff = cv.dilate(diff, None, iterations=11)

        cnts = cv.findContours(diff.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # frame_ = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        bbox_list = list()

        for cnt in cnts:
            if cv.contourArea(cnt) < self.min_area:
                continue
            (x, y, w, h) = cv.boundingRect(cnt)
            bbox_list.append((x, y, w, h))

            # cv.rectangle(frame_, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return bbox_list
    
    def draw_bbox(self, frame, bbox_list):
        frame_ = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        for (x, y, w, h) in bbox_list:
            cv.rectangle(frame_, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame_