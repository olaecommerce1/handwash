import os
import cv2
import time
import warnings
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf

#from tensorflow.keras.optimizers import Adam
#from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Time to load and print keras warnings
time.sleep(3)

# Variable to identify individual frames
global_frame_id = 1

class_names = {0:'STEP 1', 1:'STEP 2 LEFT', 2:'STEP 2 RIGHT', 3:'STEP 3', 4:'STEP 4 LEFT', 5:'STEP 4 RIGHT', 6:'STEP 5 LEFT', 7:'STEP 5 RIGHT', 8:'STEP 6 LEFT', 9:'STEP 6 RIGHT', 10:'STEP 7 LEFT', 11:'STEP 7 RIGHT'}

model = load_model("C:/Users/msila/PycharmProjects/Primary1/updated_handwash_step_model.h5")

# Image Parameters
image_height = 112
image_width = 112


def process_flow(im1, flow_vector):
    # Ensure the image has three channels
    if len(im1.shape) == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)

    hsv = np.zeros_like(im1)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return grayscale

def crop_to_height(image_array):
    height, width, channels = image_array.shape
    if height == width:
        return image_array
    image_array = np.array(image_array)
    assert height < width, "Height of the image is greater than width!"
    excess_to_crop = int((width - height) / 2)
    cropped_image = image_array[0:height, excess_to_crop:(height + excess_to_crop)]
    return cropped_image

class Frame:
    def __init__(self, pixel_values_array_input):
        global global_frame_id
        self.frame_id = global_frame_id
        global_frame_id += 1
        self.placeholder_dense_optical_flow_vector = False
        self.placeholder_pixel_values_array = pixel_values_array_input
        self.placeholder_class_predicted = -1
        self.placeholder_confidence_score = -1

    @property
    def pixel_values_array(self):
        return np.array(self.placeholder_pixel_values_array)

    @property
    def shape(self):
        return self.pixel_values_array.shape

    @property
    def frame_height(self):
        return self.pixel_values_array.shape[0]

    @property
    def frame_width(self):
        return self.pixel_values_array.shape[1]

    @property
    def number_of_channels(self):
        if len(self.pixel_values_array.shape) == 2:
            return 1
        return self.pixel_values_array.shape[2]

    @property
    def dense_optical_flow_vector(self):
        return self.placeholder_dense_optical_flow_vector

    @property
    def class_predicted(self):
        return self.placeholder_class_predicted

    @property
    def confidence_score(self):
        return self.placeholder_confidence_score

    def resize_frame(self, new_image_width, new_image_height):
        self.placeholder_pixel_values_array = cv2.resize(self.pixel_values_array, (new_image_height, new_image_width))

    def convert_to_grayscale(self):
        if self.number_of_channels > 1:
            self.placeholder_pixel_values_array = cv2.cvtColor(self.pixel_values_array, cv2.COLOR_BGR2GRAY)

    def crop_to_region(self):
        assert self.frame_height <= self.frame_width, "Height of the frame is greater than width!"
        excess_to_crop = int((self.frame_width - self.frame_height) / 2)
        cropped_image = self.pixel_values_array[0:self.frame_height, excess_to_crop:(self.frame_height + excess_to_crop)]
        self.placeholder_pixel_values_array = cropped_image
    '''
    def generate_optical_flow(self, previous_frame):
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # RGB

        previous_frame = crop_to_height(previous_frame)
        previous_frame = cv2.resize(previous_frame, (image_height, image_width))

        previous_temporal_image = previous_frame.astype(float) / 255.
        next_temporal_image = self.placeholder_pixel_values_array.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)
        temporal_image = process_flow(previous_temporal_image, temporal_image)
        self.placeholder_dense_optical_flow_vector = np.reshape(temporal_image, (1, image_width, image_height, 1))
    '''

    def generate_optical_flow(self, previous_frame):
        # Parameters for Farneback's method (these may need tuning)
        num_levels = 0
        pyr_scale = 0.5
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.2
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

        # Preprocessing the frames: cropping, resizing, converting to grayscale
        previous_frame = crop_to_height(previous_frame)
        previous_frame = cv2.resize(previous_frame, (image_height, image_width))
        next_frame = cv2.resize(self.placeholder_pixel_values_array, (image_height, image_width))

        # Convert frames to grayscale
        previous_gray = cv2.cvtColor(previous_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Compute the optical flow
        flow = cv2.calcOpticalFlowFarneback(previous_gray, next_gray, None, pyr_scale, num_levels, winsize, iterations, poly_n, poly_sigma, flags)

        # Use the flow to compute the processed image
        temporal_image = process_flow(previous_gray, flow)
        self.placeholder_dense_optical_flow_vector = np.reshape(temporal_image, (1, image_width, image_height, 1))

    def preprocess(self, final_frame_side):
        self.crop_to_region()
        self.resize_frame(final_frame_side, final_frame_side)

    def show_frame(self):
        print("Frame: " + str(self.frame_id) + " is being displayed.")
        cv2.imshow("Frame: " + str(self.frame_id), self.pixel_values_array)
        cv2.waitKey(0)
        cv2.destroyWindow("Frame: " + str(self.frame_id))

    def show_details(self):
        print("ImageID\t\t\t\t:", self.frame_id)
        print("Frame Height\t\t\t:", self.frame_height)
        print("Frame Width\t\t\t:", self.frame_width)
        print("Number of Channels\t\t:", self.number_of_channels)
        if self.class_predicted == -1:
            print("Class Predicted\t\t\t: No")
        else:
            print("Class Predicted\t\t\t:", self.class_predicted)
            print("Confidence Score\t\t:", self.confidence_score)
        if not self.dense_optical_flow_vector:
            print("Dense Optical Flow Generated\t: No")
        else:
            print("Dense Optical Flow Generated\t: Yes")

    def predict_frame1(self):
        self.convert_to_grayscale()
        spatial_image_after_reshape = np.reshape(self.placeholder_pixel_values_array, (1, image_height, image_width, 1))
        current_prediction = model.predict([np.array(spatial_image_after_reshape), np.array(self.dense_optical_flow_vector)])
        class_prediction = np.argmax(current_prediction)
        class_probability = round(current_prediction[0][class_prediction], 4)
        predicted_class = class_names[class_prediction]
        self.placeholder_class_predicted = predicted_class
        self.placeholder_confidence_score = class_probability

    def predict_frame(self):
        # Use a known good input for testing
        test_input = np.random.random((1, image_height, image_width, 1)).astype('float32')
        try:
            current_prediction = model.predict(test_input)
            print("Prediction succeeded", current_prediction)
        except Exception as e:
            print("Prediction failed:", str(e))

    def frame_predictions(self, class_predicted_input, confidence_score_input):
        self.placeholder_class_predicted = class_predicted_input
        self.placeholder_confidence_score = confidence_score_input

    def __del__(self):
        pass
