import os
import cv2
import time
import numpy as np
import frame as Frame
import frame_buffer as FrameBuffer


# Initialize a previous temporal image
previous_frame = np.array([0])

step_images = [
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done"]

class ActionSystem:
    def __init__(self):
        # Initialize the Frame Buffer with size 15
        self.frame_buffer = FrameBuffer.FrameBuffer(15)

        # Initialize the Video Stream
        self.live_stream = cv2.VideoCapture(0)
        print("Video Stream started successfully.")
        print("Changing video stream resolution..")
        self.live_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.live_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1)
        print("Video Stream resolution changed to: 1080x1080")

        # Previous frame stored to generate optical flow
        self.previous_frame = None

        # Frame counter for sampling
        self.frame_count = 1

        # Sampling rate of frames for prediction
        self.sampling_rate = 5
        self.video_fps = 30

        self.check_buffer_interval = 1000

        # Steps names (could be moved to a config file or similar)
        self.step_name = {
            "step_1": "STEP 1", "step_2_left": "STEP 2 LEFT", "step_2_right": "STEP 2 RIGHT",
            "step_3": "STEP 3", "step_4_left": "STEP 4 LEFT", "step_4_right": "STEP 4 RIGHT",
            "step_5_left": "STEP 5 LEFT", "step_5_right": "STEP 5 RIGHT",
            "step_6_left": "STEP 6 LEFT", "step_6_right": "STEP 6 RIGHT",
            "step_7_left": "STEP 7 LEFT", "step_7_right": "STEP 7 RIGHT"
        }

    def get_frame(self):
        global previous_frame
        global step_images

        success, image = self.live_stream.read()
        if not success:
            print("Failed to grab frame")
            return None

        # Frame sampling for processing
        if self.frame_count % (self.video_fps // self.sampling_rate) == 0:
            frame_object = Frame.Frame(image)

            frame_object.preprocess(112)

            frame_object.generate_optical_flow(previous_frame)

            frame_object.predict_frame()
            self.frame_buffer.add_to_buffer(frame_object)

        ret, jpeg = cv2.imencode('.jpg', image)
        previous_frame = image
        self.frame_count += 1

        # Check and report the step completed every 60 frames
        if (self.frame_count % 60 == 0  and self.frame_count > 90 ):
            step_completed = self.frame_buffer.get_step_predicted()
            return [jpeg.tobytes(), step_completed]

        return [jpeg.tobytes()]

    def get_frame_buffer_instance(self):
        return self.frame_buffer

    def check_buffer(self):
        step_completed = self.frame_buffer.get_step_predicted()

    def __del__(self):
        self.live_stream.release()
        print("Live Stream successfully released.")

