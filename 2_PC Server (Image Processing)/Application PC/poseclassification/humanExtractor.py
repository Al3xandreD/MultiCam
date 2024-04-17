import math
import time

import cv2
import numpy as np
from numpy import linalg as la

import torch
#import torch.linalg as la
import torch.nn.functional as F

from mathutils.vectors import*

from poseclassification.human import*

class HumanExtractor:
    """
    This class wrap all needed to deliver a list of 'Human' objects representing humans evolving in a scene viewed by a camera.
    This list of Human representation is estimated from a list of humans keypoints detected on the scene.
    This help to associate an identity for each pose estimations and thus not make confusion between humans detected
    on a scene (especially in case of overlapping).

    """

    global min_score
    min_score = 0

    def __init__(self, view_boundary_left=10, view_boundary_right=800, view_boundary_top=10, view_boundary_bottom=800,
                 exit_zones=None):
        self.humans = []  # List to store all humans in the scene
        self.exiting_humans = []  # List to store humans nearing the exit zones
        self.hidden_humans = []  # List to store humans in scene but currently invisible
        self.visible_humans = []  # List to store humans currently visible
        self.count = 0  # Counter for the number of humans
        self.exit_zones = exit_zones  # List of exit zones in the scene
        self.view_boundary_left = view_boundary_left  # Left boundary of the view
        self.view_boundary_right = view_boundary_right  # Right boundary of the view
        self.view_boundary_top = view_boundary_top  # Top boundary of the view
        self.view_boundary_bottom = view_boundary_bottom  # Bottom boundary of the view

    # TODO : evaluate min framerate required to make this above method work


    def update_humans(self, poses, frame):
        """
        This method aims to deliver a list of human objects from a list of humans keypoints.
        It permits to not make confusion between keypoints of several humans on a scene and conserve
        identity for all keypoints's pose estimation. This requires high enough framerate to work well
           Args:
            poses (tuple or list like): collection of poses in form of keypoints.
            deltatime (float): time passed between two frames.

        Returns:
            list of Human instances: The list of humans generated.
        """
        # print(f"HumanExtractor : Poses count : {len(poses):d}")
        # print(f"HumanExtractor : Poses type : {str(type(poses)):s}")

        self.visible_humans = []
        self.hidden_humans = []
        self.frame = frame
        self.available_poses = None
        def pickpose(human, pose):
            if human is not None and pose is not None:

                pose_index = torch.where(torch.all(self.available_poses == pose, dim=1))[0].item()
                self.available_poses = torch.cat((self.available_poses[:pose_index], self.available_poses[pose_index + 1:]), dim=0)
                human.set_pose(pose)
                self.visible_humans.append(human)
            else:
                # # print("human and/or pose is None")
                return False

        def match_pose_with_best_human(pose, humans):
            best_human = None
            previous_score = 0
            for human in humans:
                # Update human with the most looking like pose estimation
                score = self.get_matching_score(human, pose)
                if score > previous_score and score > min_score:
                    best_human = human
                    previous_score = score
            return pickpose(best_human, pose)



        if len(poses):

            # flat keypoints tensor into a matrix of pose keypoints (line = pose, column = keypoints data (x0, y0, conf0,...)
            self.available_poses = poses

            # CASE no registered humans yet : create a human for each pose detected
            if not self.count:
                for i, pose in enumerate(self.available_poses):
                    # print("taille keypoints " + str(len(pose)))
                    self.addHuman(Human(pose))
                # print(f" {len(poses):d} poses found")
                # print(f" {len(self.humans):d} humans found")

            else:
                # For each pose associate it with the best human it maches only if score above a certain amount
                for i, pose in enumerate(self.available_poses):
                    if match_pose_with_best_human(pose, self.humans):
                        pose_index = torch.where(torch.all(self.available_poses == pose, dim=1))[0].item()
                        # Remove pose tensor from available poses
                        self.available_poses = torch.cat((self.available_poses[:pose_index], self.available_poses[pose_index + 1:]),
                                                    dim=0)


                # For remaining poses that not was associated with a human in the previous step -> new humans appeared
                for pose in self.available_poses:
                    # print("Adding human")
                    self.addHuman(Human(pose))

                # Remove humans who exited the scene
                for human in self.exiting_humans:
                    # print("exiting", human.acceleration)
                    # Check if human pose has been updated. If not, we presume the human has exited the scene
                    if human not in self.visible_humans:
                        pass
                        #self.removeHuman(human)

                # Determine hidden humans
                for human in self.humans:

                    # Check if human has been updated as visible. If not, we presume the human is hidden
                    if human not in self.visible_humans:
                        self.hidden_humans.append(human)

            # At the end call update for all humans and mark those who are exiting the scene
            for human in self.humans:
                human.update()
                if self.is_exiting(human):
                    self.exiting_humans.append(human)

    def get_matching_score(self, human, pose):
        weights = [1, 1, 0, 0]
        min_kpt_score = 0.4

        # Human pose : separate keypoints positions and keypoints confidence
        human_kpts_xy = human.get_pose().reshape(-1, 3)[:, :-1]
        human_kpts_scores = human.get_pose().reshape(-1, 3)[:, -1]

        # Given pose : separate  and keypoints confidence
        kpts_xy = pose.reshape(-1, 3)[:, :-1]
        kpts_scores = pose.reshape(-1, 3)[:, -1]

        # Consider only keypoints coordinates whose conf > min conf:
        indices = torch.where((human_kpts_scores > 0.3) & (kpts_scores > 0.3))[0]

        # => keypoints positions whose conf > min conf
        kpts1 = human_kpts_xy[indices]
        kpts2 = kpts_xy[indices]

        # TODO hand calculus to determine weight value etc...Protocol :
        # Measure distance value between two poses when drop of mean score (2%)
        # Measure max distance for one person moving
        # calcul...




        # keypoints euclid distance based score
        pos_score = 1 / (euclidean_distance(kpts1, kpts2)/8 + 1)
        # print("POSE SCORE", pos_score)



        # center distance based score
        center1 = human.center
        center2 = kpts_xy.mean(dim=0)
        centers_distance = euclidean_distance(center1, center2)
        center_score = 1 / (0.1 * centers_distance + 1)

        # keypoints correl based score
        correl_score = correlation(kpts1, kpts2)
        # TODO ...
        conf_score = 0

        # color histogram based score
        col_hist_score = 0 #correlation(human.get_color_histogram(), get_color_histogram(pose, frame))

        score = (weights[0] * pos_score + weights[1] * correl_score + center_score * weights[2] + col_hist_score * weights[3])/sum(weights)
        # print("matching score", score)
        return score

    def is_exiting(self, human):
        # Check if near view boundaries
        res = (human.center[0] < self.view_boundary_left or
               human.center[0] > self.view_boundary_right or
               human.center[1] < self.view_boundary_top or
               human.center[1] > self.view_boundary_bottom)

        # Check if near an exit zone TODO ...
        # for exit_zone in self.exit_zones:
        #     pass
        return res

    def addHuman(self, human):
        self.humans.append(human)
        self.count += 1

    def removeHuman(self, human):
        self.humans.remove(human)
        self.count -= 1



