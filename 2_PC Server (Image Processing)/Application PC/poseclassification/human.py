import time
import numpy as np

import torch
import torch.linalg as la


# Pose estimation --> Humans extraction --> Activity recognition for each human

# Keypoints structure : tensor[x_coord0, y_coord0, conf0,  x_coord1, y_coord1, conf1, ...]


class Human:
    def __init__(self, keypoints=None, pose_buffer_size=5):
        self.old_speed = torch.zeros((1, 2), device="cuda")
        self.id = 0
        self.current_keypoints = keypoints
        self.ticks = 0
        self.pose_buffer_size = pose_buffer_size

        self.center = None
        self.width, self.height = None, None
        self.xyxy_buff = []
        self.kpts_buff = []
        self.w_h_ratio = -1
        self.global_abs_speed = torch.zeros((1, 1), device="cuda")
        self.last_not_zero_speed = -1
        self.last_time_not_zero_speed = time.time()
        self.idle_time = -1
        self.acceleration = 0
        self.avg_acceleration = -1
        self.labels = ["unused", "dynamique", "immobile", "choc"]

        self.speed = None  # Speed (pixels/s)   -   TODO : Introduce speed in m/s
        self.map_position = None  # position in map (m)
        self.floor_position = None  # floor position
        self.limb_speed = None
        self.pose_index = None
        self.pose_name = None
        self.set_pose(keypoints)

    def get_rect_geom(self, bbox):
        tl, br = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        width = br[0] - tl[0]
        height = br[1] - tl[1]
        # center = np.array([[(br[0] + tl[0]) / 2], [(br[1] + tl[1]) / 2]])
        center = self.current_keypoints.reshape(-1, 3)[:, :-1].mean(dim=0)
        print(center)
        return width, height, center, tl, br

    def bufferize(self, kpts, bbox):
        # "Buffer" (useful for smoothed speeds computing)
        if len(self.xyxy_buff) < self.pose_buffer_size:
            self.xyxy_buff.append(bbox)
            self.kpts_buff.append(kpts)
        else:
            self.xyxy_buff.pop(0)
            self.xyxy_buff.append(bbox)
            self.kpts_buff.pop(0)
            self.kpts_buff.append(kpts)

    def set_pose(self, keypoints):
        """
        Set current pose
           Args:
            pose : list of keypoints representing a human.
        """
        self.current_keypoints = keypoints

    def get_pose(self, pose_index=-1):
        """
        Get current pose
           Args:
            pose_index : index of the poses buffer. If out ouf bound or -1 return the last pose
        """
        if len(self.kpts_buff):
            max_ind = len(self.kpts_buff) - 1
            if pose_index > max_ind or pose_index < 0:
                pose_index = max_ind
            return self.kpts_buff[pose_index]
        else:
            return None

    def update(self, keypoints=None, dt=0.2):

        if keypoints is not None:
            self.set_pose(keypoints)

        # Get current pose (last) for further calculations:
        curr_pose = self.current_keypoints
        if curr_pose is not None:

            # Separate keypoints positions and keypoints confidence
            kpts_xy = curr_pose.reshape(-1, 3)[:, :-1]
            kpts_score = curr_pose.reshape(-1, 3)[:, :-1]

            # Compute bounding box
            self.bbox = [min(kpts_xy[:, 0]), min(kpts_xy[:, 1]), max(kpts_xy[:, 0]),
                         max(kpts_xy[:, 1])]

            # "Bufferize" pose
            self.bufferize(curr_pose, self.bbox)

            # TODO improve measures by normalizing according to human dims so that they are independents to distance
            # Get additional Bounding box geometry infos (width, height, center, area...)
            self.width, self.height, self.center, _, _ = self.get_rect_geom(self.bbox)
            self.w_h_ratio = self.width / self.height if self.height > 0 else 0
            # print(f'width:{width:2f}')
            # print(f'height:{height:2f}')
            # print(f'ratio:{w_h_ratio:2f}')

            speed = torch.zeros((1, 2), device="cuda")
            self.old_speed = torch.zeros((1, 1), device="cuda")
            steps = 3
            self.kpts_xy_speeds = torch.zeros_like(kpts_score)
            self.kpts_xy_avg_speed = torch.zeros((1, 1), device="cuda")
            # TODO optimize calculus :

            old_center = torch.zeros((1, 2), device="cuda")
            if len(self.xyxy_buff) == self.pose_buffer_size:
                for i in range(1, len(self.xyxy_buff)):
                    # Get Bounding box global speed
                    _, _, center, _, _ = self.get_rect_geom(self.xyxy_buff[i])
                    #_, _, old_center, _, _ = self.get_rect_geom(self.xyxy_buff[i - 1])
                    speed += abs(center - old_center)  # mean filter
                    old_center = center

                    # Get Limbs/Keypoints speed
                    curr_kpts = self.kpts_buff[i].unfold(0, 2, 1)[::steps]
                    old_kpts = self.kpts_buff[i - 1].unfold(0, 2, 1)[::steps]
                    self.kpts_xy_speeds += abs(curr_kpts - old_kpts)/17

                speed /= self.pose_buffer_size * dt
                self.global_abs_speed = la.norm(speed)

                self.kpts_xy_speeds /= self.pose_buffer_size * dt
                self.kpts_speeds_norms = torch.norm(self.kpts_xy_speeds, dim=1)
                # print(kpts_speeds_norm)
                self.kpts_xy_avg_speed = self.kpts_speeds_norms.mean()
                # print("Average keypoints speed", self.kpts_xy_avg_speed)

                # Get bbox global acceleration
                self.acceleration = (self.global_abs_speed - self.old_speed) / dt

                self.old_speed = self.global_abs_speed
                # print(acceleration)


                # Get bbox global avg acceleration
                n = self.ticks - self.pose_buffer_size + 1
                self.avg_acceleration = ((n - 1) * self.avg_acceleration + self.acceleration) / n

                self.avg_acceleration += self.acceleration / self.pose_buffer_size * dt


                # print("Acceleration", self.avg_acceleration)

            # Get Idle time
            idle_speed_thres = 15
            if self.global_abs_speed < idle_speed_thres:
                self.idle_time = time.time() - self.last_time_not_zero_speed
            else:
                self.last_not_zero_speed = self.global_abs_speed
                self.last_time_not_zero_speed = time.time()
                self.idle_time = 0
            # print("Last non-zero speed", self.last_not_zero_speed)
            # print(f'idle time:{self.idle_time:2f}')
