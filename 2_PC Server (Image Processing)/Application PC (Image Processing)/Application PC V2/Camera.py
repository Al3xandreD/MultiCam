import time
from threading import Thread

import cv2

from utils.urlconnection.urlconnect import *
import numpy as np


def gcd_resize(input_frame, gcd):
    frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
    if gcd != 1:
        adapted_width = gcd * (frame_width // gcd)
        adapted_height = gcd * (frame_height // gcd)
        adapted_frame = cv2.resize(input_frame, (adapted_width, adapted_height))
    else:
        adapted_frame = input_frame
    return adapted_frame


def rotate_image(image, angle):
    # Get the image center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotated_image = image

    if (angle != 0):
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def generate_grid_image(width, height, tile_size, col1=0, col2=255):
    # Grid pattern generation (Meth 1)
    # grid_pattern = np.eye(2) * 255
    # grid_pattern = cv2.resize(grid_pattern, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

    # Grid pattern generation (Meth 2)
    grid_pattern = np.block([[col1 * np.ones((tile_size, tile_size))], [col2 * np.ones((tile_size, tile_size))]])
    grid_pattern = np.block([grid_pattern, np.flip(grid_pattern, axis=0)])
    return np.tile(grid_pattern, (height // (2 * tile_size), width // (2 * tile_size)))


class UrlCamera:
    W_DEFAULT, H_DEFAULT = 640, 480
    w, h = W_DEFAULT, H_DEFAULT

    def __init__(self, source, source_is_auto_refresh=False, pair_cam=None, is_ref=False, no_frame_timeout=1,
                 frame_time=0, frames_skip=0,
                 frame_width=W_DEFAULT,
                 frame_height=H_DEFAULT, channels_count=3, flip_code=2, rot_angle=1, ):
        """

        0: Flip vertically(upside down).
        1: Flip horizontally(left to right).
        -1: Flip both vertically and horizontally(upside down and left to right).

        """
        self.started = False
        self.success = False
        self.flip_code = flip_code
        self.rot_angle = rot_angle
        self.source = source
        self.pair_cam = pair_cam
        self.is_ref = is_ref
        self.frame_width, self.frame_height = frame_width, frame_height
        self.channels_count = channels_count
        self.no_frame_timeout = no_frame_timeout
        self.frame_time = frame_time
        self.frames_skip = frames_skip
        self.output_frame = gcd_resize(
            rotate_image(cv2.flip(generate_grid_image(self.frame_width, self.frame_height, 32), self.flip_code),
                         self.rot_angle), 64)
        self.source_is_auto_refresh = source_is_auto_refresh
        self.idle_frame = self.output_frame
        self.is_webcam = False
        self.webcam = None

    def __apply_transform(self, frame):
        flipped_frame = cv2.flip(frame, self.flip_code)
        rotated_frame = rotate_image(flipped_frame, self.rot_angle)
        resized_frame = gcd_resize(rotated_frame, 64)
        return resized_frame

    def update(self, url):
        # Read next stream frame in a daemon thread
        f = 0
        last_success_frame_time_capture = 0
        self.started = True
        while self.started:
            frame_cv = None
            self.success = False
            f += 1
            if f == self.frames_skip + 1:
                f = 0
                if self.is_webcam:
                    self.success, frame_cv = self.webcam.read()
                else:
                    # Getting image from url (Meth 1)
                    try:
                        if url_ok(url, 1):
                            frame_request = urllib.request.urlopen(url)
                            frame_np = np.array(bytearray(frame_request.read()), dtype=np.uint8)
                            frame_cv = cv2.imdecode(frame_np, -1)
                            self.success = frame_cv is not None

                    except:
                        # TODO : propage exception to instance interface
                        print(f'failed to open cam n°{0:d}')

                if not self.success:
                    """
                    If no frame is delivered within timeout return grid screen

                    """
                    if time.time() - last_success_frame_time_capture > self.no_frame_timeout:
                        # Idle frame
                        frame_cv = generate_grid_image(self.frame_width, self.frame_height, 32)  # Grid
                        # frame_cv = np.ones((self.frame_height, self.frame_width, self.channels_count)) * 255 #Solid color

                else:
                    last_success_frame_time_capture = time.time()
                    # Get real shape
                    self.frame_height, self.frame_width = frame_cv.shape[0], frame_cv.shape[1]

                # Apply transforms corrections and update output frame if not null
                if frame_cv is not None:
                    self.output_frame = self.__apply_transform(frame_cv)
                time.sleep(self.frame_time)  # wait time

    ########## Prepare and start camera thread #########
    def start(self):
        if type(self.source) == int or self.source_is_auto_refresh:
            self.is_webcam = True
            self.webcam = cv2.VideoCapture(self.source)
        thread = Thread(target=self.update, args=([self.source]), daemon=True)
        thread.start()

    def stop(self):
        self.started = False
        if self.is_webcam:
            self.webcam.release()


    def read(self):
        return self.success, self.output_frame
