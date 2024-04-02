import time
import numpy as np
#Pose estimation --> Humans --> Pose classification


class Humans:
    def __init__(self):
        self.humans = []
        self.count = 0
    def update_humans(self, all_keypoints):
        if len(all_keypoints):
            #If no humans loop through poses
            if not self.count:
                for keypoints in enumerate(all_keypoints):
                    human = Human(self, keypoints)
                    self.humans.append(human)
            else:
                # Update human with the most looking like pose estimation
                print()


class Human:
    def __init__(self, keypoints=None, pose_buffer_size=5):
        self.current_keypoints = keypoints
        self.ticks = 0
        self.pose_buffer_size = pose_buffer_size

        self.xyxy_buff = []
        self.kpts_buff = []
        self.w_h_ratio = -1
        self.global_abs_speed = -1
        self.last_not_zero_speed = -1
        self.last_time_not_zero_speed = time.time()
        self.idle_time = -1
        self.acceleration = 0
        self.avg_acceleration = -1
        self.labels = ["unused", "dynamique", "immobile", "choc"]

        self.speed # Speed (pixels/s)   -   TODO : Introduce speed in m/s
        self.map_position # position in map (m)
        self.floor_position # floor position
        self.limb_speed
        self.pose_index
        self.pose_name


    def get_rect_geom(self, xyxy):
        tl, br = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        width = br[0] - tl[0]
        height = br[1] - tl[1]
        center = np.array([[(br[0] + tl[0]) / 2],
                           [(br[1] + tl[1]) / 2]])
        return width, height, center, tl, br

    def bufferize(self, kpts, xyxy):
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

    def update(self, keypoints):

        # "Bufferize" pose
        self.kpts_buff.append(keypoints)

