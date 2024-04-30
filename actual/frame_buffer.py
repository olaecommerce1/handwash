import os
import cv2
import time
from collections import defaultdict
import warnings
import numpy as np

class FrameBuffer:
    @property
    def prediction_sampling(self):
        return self.placeholder_prediction_sampling

    @property
    def frame_buffer_array(self):
        return self.placeholder_frame_buffer_array

    def __init__(self, buffer_size_input):
        self.buffer_size = buffer_size_input
        self.frame_buffer_pointer = 0
        self.placeholder_frame_buffer_array = [[-99, -99, -99]] * buffer_size_input
        self.placeholder_prediction_sampling = 0
        print(f"Frame Buffer of size {self.buffer_size} created successfully.")

    def add_to_buffer(self, frame_object):
        self.placeholder_frame_buffer_array[self.frame_buffer_pointer] = frame_object

        # Update the frame_buffer_pointer in a circular manner
        self.frame_buffer_pointer = (self.frame_buffer_pointer+1)%self.buffer_size

    def show_buffer(self):
        print("The Frame Buffer is currently: ", end=" ")
        for frame_buffer_iterator in range(len(self.frame_buffer_array)):
            try:
                if (self.frame_buffer_array[frame_buffer_iterator][0] == -99):
                    print("[Empty Buffer Slot]", end="")
            except:
                print([self.frame_buffer_array[frame_buffer_iterator].frame_id,
                       self.frame_buffer_array[frame_buffer_iterator].class_predicted,
                       self.frame_buffer_array[frame_buffer_iterator].confidence_score], end="")
            # Continue printing commas unless you are printing the
            # last frame_object in the Frame Buffer
            if (frame_buffer_iterator != (len(self.frame_buffer_array) - 1)):
                print(",", end=" ")
        # Print a newline in the end, to beautify the output
        print(" ")

    def set_prediction_sampling(self, prediction_sampling_input):
        assert (prediction_sampling_input <= self.buffer_size), "The number of prediction sampling frames is more than the buffer size!"

        self.placeholder_prediction_sampling = prediction_sampling_input

    def get_step_predicted(self):
        steps = defaultdict(lambda: 0)

        for prediction in self.frame_buffer_array:
            steps[prediction.class_predicted] += 1

        print("STEP PREDICTED:",max(steps, key=steps.get))
        return max(steps, key=steps.get)

    def clear_buffer(self):
        self.frame_buffer_array = [[-99, -99, -99]] * self.buffer_size

    def __del__(self):
        print(f"Frame Buffer of size {self.buffer_size} successfully deleted.")

