import math
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import output_to_keypoint
from numpy import linalg as la

"""This class permit to classify a pose (associate a given pose to a known posture label)"""


# Symbolic approach :
class SymbolicPoseClassifier:
    def __init__(self, labels_matrix=None, thresholds_matrix=None, pose_buffer_size=5):

        self.ticks = 0
        self.pose_buffer_size = pose_buffer_size

        inf = math.inf
        self.thresholds_mat = np.array(
            [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
             [[0, 0], [60, inf], [0, 0], [0, 0], [0, 0]],
             [[0, 0], [0, 15], [0, 0], [0, 0], [10, inf]],
             [[0, 0], [0, 0], [20, inf], [60, 0], [3, inf]],
             ])

        # Measures / Observations
        self.xyxy_buff = []
        self.kpts_buff = []
        self.w_h_ratio = -1
        self.global_abs_speed = -1
        self.last_not_zero_speed = -1
        self.last_time_not_zero_speed = time.time()
        self.idle_time = -1
        self.acceleration = 0
        self.avg_acceleration = -1
        self.kpts_xy_speeds = None

        self.labels = ["unused", "dynamique", "immobile", "choc"]

    def compare(self, measures_vec, thresholds_mat):
        # TODO : improve decision
        matches_counts = np.sum((measures_vec > thresholds_mat[:, :, 0]) * (measures_vec < thresholds_mat[:, :, 1]), axis=1)
        label_id = np.argmax(matches_counts)
        label = self.labels[label_id]
        print(label)
        return label_id


    def get_rect_geom(self, xyxy):
        tl, br = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        width = br[0] - tl[0]
        height = br[1] - tl[1]
        center = np.array([[(br[0] + tl[0]) / 2],
                           [(br[1] + tl[1]) / 2]])
        return width, height, center, tl, br

    def get_measures(self, kpts, xyxy):
        # "Buffer" (useful for smoothed speeds computing)
        if len(self.xyxy_buff) < self.pose_buffer_size:
            self.xyxy_buff.append(xyxy)
            self.kpts_buff.append(kpts)
        else:
            self.xyxy_buff.pop(0)
            self.xyxy_buff.append(xyxy)
            self.kpts_buff.pop(0)
            self.kpts_buff.append(kpts)

        # TODO improve measures by normalizing according to human dims so that they are independents to distance

        # Get Bounding box ratio
        width, height, center, _, _ = self.get_rect_geom(xyxy)
        self.w_h_ratio = width / height if height > 0 else 0
        # print(f'width:{width:2f}')
        # print(f'height:{height:2f}')
        # print(f'ratio:{w_h_ratio:2f}')

        # TODO calculate dt, arbitrary value for the moment :
        dt = 0.2
        speed = np.zeros((2, 1))
        old_speed = 0
        steps = 3
        num_kpts = len(kpts) // steps
        # arr = self.kpts_buff[0].detach().cpu().numpy()
        self.kpts_xy_speeds = torch.zeros_like(self.kpts_buff[0].unfold(0, 2, 1)[::steps])

        # TODO optimize calculus :

        if len(self.xyxy_buff) == self.pose_buffer_size:

            for i in range(1, len(self.xyxy_buff)):
                # Get Bounding box global speed
                _, _, center, _, _ = self.get_rect_geom(self.xyxy_buff[i])
                _, _, old_center, _, _ = self.get_rect_geom(self.xyxy_buff[i - 1])
                speed += abs(center - old_center)  # mean filter

                # Get Limbs/Keypoints speed
                curr_kpts = self.kpts_buff[i].unfold(0, 2, 1)[::steps]
                old_kpts = self.kpts_buff[i - 1].unfold(0, 2, 1)[::steps]
                kpts_xy_speeds += abs(curr_kpts - old_kpts)

            speed /= self.pose_buffer_size * dt
            self.global_abs_speed = la.norm(speed)
            print(self.last_not_zero_speed)

            kpts_xy_speeds /= self.pose_buffer_size * dt
            self.kpts_speeds_norms = torch.norm(kpts_xy_speeds, dim=1)
            # print(kpts_speeds_norm)
            self.kpts_xy_avg_speed = self.kpts_speeds_norms.mean()
            # print(kpts_xy_avg_speed)

            # Get bbox global acceleration
            acceleration = (self.global_abs_speed - old_speed) / dt
            old_speed = self.global_abs_speed
            # print(acceleration)

            # Get bbox global avg acceleration
            n = self.ticks - self.pose_buffer_size + 1
            self.avg_acceleration = ((n - 1) * self.avg_acceleration + acceleration) / n
            print(self.acceleration)

        # Get Idle time
        idle_speed_thres = 15
        if self.global_abs_speed < idle_speed_thres:
            self.idle_time = time.time() - self.last_time_not_zero_speed
        else:
            self.last_not_zero_speed = self.global_abs_speed
            self.last_time_not_zero_speed = time.time()
            self.idle_time = 0
        # print(f'idle time:{self.idle_time:2f}')

        # Get global score

        return np.array([self.w_h_ratio, self.global_abs_speed, self.last_not_zero_speed, self.acceleration, self.idle_time])

    def symbolic_classify_core(self, xyxy, kpts, conf, pose_buffer_size=5):

        self.ticks += 1

        # Step 1 : Measure all characteristics
        self.measures_vec = self.get_measures(kpts, xyxy)

        # Step 2 : Compare characteristics values matrix with characteristics thresholds matrix :
        decision = self.compare(self.measures_vec, self.thresholds_mat)

        # return labelId

    def symbolic_classify(self, detection_results):

        try:
            # Extract pose keypoints
            for i, pose in enumerate(
                    detection_results):  # detections per image #TODO : Try understand this line better (seems to be several version of same detection results)

                if len(detection_results):  # check if no pose
                    for c in pose[:, 5].unique():  # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        # print("No of Objects in Current Frame : {}".format(n))

                    for det_index, (*xyxy, conf, cls) in enumerate(
                            reversed(pose[:, :6])):  # loop over poses for drawing on frame
                        kpts = pose[det_index, 6:]

                        # Classify pose
                        self.symbolic_classify_core(xyxy, kpts, conf, pose_buffer_size=self.pose_buffer_size)
        except Exception as e:
            print(f"An error occurred: {e}")

        return 1


# Machine Learning approach :
# see https://docs.nvidia.com/tao/tao-toolkit/text/pose_classification/pose_classification.html
# https://github.com/mmakos/HPC


class MLPoseClassifier:
    def __init__(self):
        print()
