import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, Sequential

# Disable GPU usage explicitly
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Time delay simulation for loading or other operations
import time

# time.sleep(3)

global_frame_id = 1

class_names = {0: 'STEP 1', 1: 'STEP 2 LEFT', 2: 'STEP 2 RIGHT', 3: 'STEP 3', 4: 'STEP 4 LEFT', 5: 'STEP 4 RIGHT',
               6: 'STEP 5 LEFT', 7: 'STEP 5 RIGHT', 8: 'STEP 6 LEFT', 9: 'STEP 6 RIGHT', 10: 'STEP 7 LEFT',
               11: 'STEP 7 RIGHT'}

image_height = 112
image_width = 112


def process_flow1(im1, flow_vector):
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    h, s, grayscale = cv2.split(hsv)

    return (grayscale)

def process_flow(im1, flow_vector):
    # Create an HSV image with the same spatial dimensions as im1 but explicitly with 3 channels
    hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255  # Set the Hue
    hsv[:, :, 1] = 255  # Set the Saturation
    mag, ang = cv2.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Convert angle to degrees for Hue
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude for Value
    # Convert HSV to grayscale to extract the Value channel
    h, s, grayscale = cv2.split(hsv)
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


model = load_model("C:/Users/msila/PycharmProjects/Primary1/updated_handwash_step_model.h5")


class Frame:
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

    def __init__(self, pixel_values_array_input):
        global global_frame_id
        self.frame_id = global_frame_id
        global_frame_id += 1
        self.placeholder_dense_optical_flow_vector = False
        self.placeholder_pixel_values_array = pixel_values_array_input
        self.placeholder_class_predicted = -1
        self.placeholder_confidence_score = -1

    def resize_frame(self, new_width, new_height):
        self.placeholder_pixel_values_array = np.array(cv2.resize(self.pixel_values_array, (new_width, new_height)))

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

    def preprocess1(self, final_frame_side):
        # Example preprocessing steps
        self.convert_to_grayscale()
        self.resize_frame(final_frame_side, final_frame_side)

    def convert_to_grayscale(self):
        assert (self.number_of_channels > 1), "The image to convert_to_grayscale is already Grayscale"
        self.placeholder_pixel_values_array = cv2.cvtColor(self.pixel_values_array, cv2.COLOR_BGR2GRAY)

    def crop_to_region(self):
        assert self.frame_height != self.frame_width, "Frame is already a cropped to region of interest."
        # assert self.pixel_values_array.shape[0] != self.pixel_values_array.shape[1], "Frame is already a cropped to region of interest."
        assert self.frame_height <= self.frame_width, "Height of the frame is greater than width!"
        # assert self.pixel_values_array.shape[0] <= self.self.pixel_values_array.shape[1], "Height of the frame is greater than width!"

        excess_to_crop = int((self.frame_width - self.frame_height) / 2)
        cropped_image = self.pixel_values_array[0:self.frame_height,
                        excess_to_crop:(self.frame_height + excess_to_crop)]

        # excess_to_crop = int((self.pixel_values_array.shape[1] - self.pixel_values_array.shape[0])/2)
        # cropped_image = self.pixel_values_array[0:self.pixel_values_array.shape[0], excess_to_crop:(self.pixel_values_array.shape[0]+excess_to_crop)]

        self.placeholder_pixel_values_array = cropped_image

    def generate_optical_flow12(self, previous_frame):

        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        assert (self.dense_optical_flow_vector == False), "Optical Flow already generated for the current frame."

        # Cropping the previous_frame
        previous_frame = crop_to_height(previous_frame)

        # Resizing the previous frame
        previous_frame = cv2.resize(previous_frame, (image_height, image_width))

        # Numpy Arrays
        previous_temporal_image = previous_frame.astype(float) / 255.
        next_temporal_image = self.placeholder_pixel_values_array.astype(float) / 255.

        # previous_temporal_image = previous_temporal_image.astype(np.float32)
        # next_temporal_image = next_temporal_image.astype(np.float32)

        # print("ACTUAL:")
        # print(len(self.pixel_values_array),"x",len(self.pixel_values_array[0]), "x", len(self.pixel_values_array[0][0][0]))

        # Adding new method
        #u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth,
                                             #nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        #temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)
        #temporal_image = process_flow(previous_temporal_image, temporal_image)

        # temporal_image_grayscale = cv2.cvtColor(temporal_image, cv2.COLOR_BGR2GRAY)
        #temporal_image_after_reshape = np.reshape(temporal_image, (1, image_width, image_height, 1))

        #self.placeholder_dense_optical_flow_vector = temporal_image_after_reshape

    def generate_optical_flow(self, previous_frame):
        assert (not self.dense_optical_flow_vector), "Optical Flow already generated for the current frame."

        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        #previous_frame = crop_to_height(previous_frame)
        previous_frame = cv2.resize(previous_frame, (image_width, image_height))

        #previous_temporal_image = previous_frame.astype(float) / 255.
        #next_temporal_image = self.placeholder_pixel_values_array.astype(float) / 255.

        # Prepare the current frame
        current_frame = cv2.cvtColor(self.placeholder_pixel_values_array,
                                     cv2.COLOR_BGR2GRAY) if self.placeholder_pixel_values_array.ndim == 3 else self.placeholder_pixel_values_array
        current_frame = cv2.resize(current_frame, (image_width, image_height))

        #current_frame = cv2.resize(self.placeholder_pixel_values_array, (image_width, image_height))
        #if len(previous_frame.shape) == 3:
        #    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        #if len(current_frame.shape) == 3:
        #    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        #print("ACTUAL:")
        #print(self.pixel_values_array,"x",self.pixel_values_array[0], "x", self.pixel_values_array[0][0][0])

        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #temporal_image = np.concatenate((magnitude, angle), axis=2)
        temporal_image = np.concatenate((magnitude[..., None], angle[..., None]), axis=2)
        processed_flow = process_flow(current_frame, temporal_image)

        #mask = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        self.placeholder_dense_optical_flow_vector = np.reshape(processed_flow, (1, image_height, image_width, 1))

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

    def predict_frame(self):
        self.convert_to_grayscale()
        spatial_image_after_reshape = np.reshape(self.placeholder_pixel_values_array, (1, image_height, image_width, 1))
        current_prediction = model.predict([np.array(spatial_image_after_reshape), np.array(self.dense_optical_flow_vector)])
        class_prediction = np.argmax(current_prediction)
        class_probability = round(current_prediction[0][class_prediction], 4)
        predicted_class = class_names[class_prediction]
        self.placeholder_class_predicted = predicted_class
        self.placeholder_confidence_score = class_probability

    def get_frame_id(self):

        return (self.frame_id)

    def frame_predictions(self, class_predicted_input, confidence_score_input):

        self.placeholder_class_predicted = class_predicted_input
        self.placeholder_confidence_score = confidence_score_input

    def __del__(self):
        pass