import numpy as np

class MotionDetector:
    def __init__(self, resolution = (640, 480), forget_percentage = 50, cumsum_enabled = True, delta_time = 2) -> None:
        self.cumsum_enabled = cumsum_enabled
        self.delta_time = delta_time
        self.resolution = resolution
        self.forget_percentage = forget_percentage
        self.frame_difference = np.empty(shape=[ 3, *resolution ])
        self.prev_frame = np.empty(shape=[ 3, *resolution ])
        self.frame_cumsum = np.zeros(shape=[ 3, *resolution ])


    ### set time difference in frames (for subtraction)
    def set_delta_time(self, new_delta_time):
        self.delta_time = new_delta_time

    def set_cumsum_mode(self, cumsum_mode):
        self.cumsum_enabled = cumsum_mode

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.frame_difference = np.empty(shape=[ 3, *resolution ])
        self.prev_frame = np.empty(shape=[ 3, *resolution ])
        self.frame_cumsum = np.zeros(shape=[ 3, *resolution ])

    def get_difference(self, frame: np.ndarray):
        if frame.shape != [ 3, *self.resolution ]:
            raise SystemExit("Resolution of detector and image doesn't match")
        
        self.frame_difference = frame - self.prev_frame
        output = self.frame_difference

        if self.cumsum_enabled:
            self.frame_cumsum = self.frame_cumsum * (self.forget_percentage / 100) + self.frame_difference
            output = self.frame_cumsum

        return output       