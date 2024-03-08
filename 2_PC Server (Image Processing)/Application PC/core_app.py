# Ref : https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
from os.path import dirname, abspath
import cv2
import openpyxl
import argparse
from utils.plots import colors, plot_one_box_kpt
from camera import GenericCamera
from poseestimation.poseestimator import PoseEstimator
from poseclassification.poseclassifier import SymbolicPoseClassifier

from network.localnetwork import *

#T
common_resources_dir = "\#Common Resources"
root_folder_content_signature = common_resources_dir  # To find root folder automatically

# EXCEL IP CAMS DATABASE
IPCAMS_DATABASE = common_resources_dir + '\All_IPCams.xlsx'
BAT_ID = 1
IPCAMS_DATABASE_STARTROW = 3
IPCAMS_DATABASE_STARTCOL = 6

# CAMS
W_DEFAULT, H_DEFAULT = 800, 600  # 768, 576
w, h = W_DEFAULT, H_DEFAULT

  


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


########### Pose drawer ###############
def draw_poses(frame, detection_results, names, line_thickness, hide_labels, hide_conf):
    # frame = frame[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    # frame = frame.cpu().numpy().astype(np.uint8)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
    # gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i, pose in enumerate(
            detection_results):  # detections per image #TODO To clarify better (seems to be several version of same detection results)

        if len(detection_results):  # check if no pose
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                print("No of Objects in Current Frame : {}".format(n))

            for det_index, (*xyxy, conf, cls) in enumerate(
                    reversed(pose[:, :6])):  # loop over poses for drawing on frame
                c = int(cls)  # integer class
                kpts = pose[det_index, 6:]
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box_kpt(xyxy, frame, label=label, color=colors(c, True),
                                 line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                 orig_shape=frame.shape[:2])
    return frame


def ndarray_info(nda):
    print("type\t: ", nda.dtype, "\nshape\t: ", nda.shape, "\nmin\t: ", nda.min(), "\nmax\t: ", nda.max(), "\n")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')  # device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # boxhideconf
    opt = parser.parse_args()
    return opt


class Main:

    def __init__(self, opt_arg):
        ################## Get root project directory (Only for development purposes) ##################
        i = 0
        root_project_dir = dirname(abspath(__file__))
        while not os.path.exists(root_project_dir + root_folder_content_signature):
            root_project_dir = dirname(abspath(root_project_dir))
            i += 1
            assert i < 100, f'Root project directory not found. Please ensure that there is a "\{root_folder_content_signature}" folder inside it'

        root_project_dir += "\""
        print(f'Root project directory found : \n {root_project_dir}')
        #################################################################

        ################# Get options ###############
        self.opt = opt_arg
        print(f"opt.source: {self.opt.source:s}")
        source = int(self.opt.source)  # 0 : Test with PC webcam / !=0 : Use url cameras

        ############### Get LAN Infos ###############
        _, self.lan_ssid = get_current_lan_ssid()

        ##############  Init cameras video stream ###############
        self.cameras = []
        if source != 0:
            # Get Cams Urls
            IPCAMS_DATABASE_PATH = root_project_dir + IPCAMS_DATABASE
            IPCAMS_DATABASE_PATH = IPCAMS_DATABASE_PATH.replace("\"", "\\")  # Convert to double backslashes if needed
            wb = openpyxl.load_workbook(IPCAMS_DATABASE_PATH, data_only=True)
            bat_names = wb.sheetnames
            ws = wb[bat_names[BAT_ID]]
            self.urls = [ws.cell(row=i + IPCAMS_DATABASE_STARTROW, column=IPCAMS_DATABASE_STARTCOL).value for i in
                         range(1, ws.max_row + 1) if
                         ws.cell(row=i + IPCAMS_DATABASE_STARTROW, column=IPCAMS_DATABASE_STARTCOL).value is not None]
            # self.urls = self.urls[:1]
            print(self.urls)
            n = len(self.urls)
            rot_angles = [90 * 0 for i in range(n)]  # TODO rm 0
            rot_angles[3] = 0

            for i, url in enumerate(self.urls):
                camera = GenericCamera(url, rot_angle=rot_angles[i], source_is_auto_refresh=(IPCAMS_DATABASE_STARTCOL == 4))
                camera.start()
                self.cameras.append(camera)

        else:
            camera = GenericCamera(0, rot_angle=0)
            camera.start()
            self.cameras.append(camera)

        ###########################################################

        ################## Init pose estimator ###################
        self.pose_estimator = PoseEstimator(self.opt.device)
        self.names = self.pose_estimator.names
        ###########################################################

        ################## Init pose classifier ###################
        self.pose_classifier = SymbolicPoseClassifier()
        ###########################################################

    def update(self):
        j = 0
        n = len(self.cameras)
        output = [None for i in range(n)]
        for i, cam in enumerate(self.cameras):
            success, frame = cam.read()

            # Detection
            enable_detection = 1
            success = 1
            enable_pose_drawing = 1
            if success:
                if enable_detection:
                    if i == j:
                        yol_frame, results, _, _ = self.pose_estimator.detect(frame)
                        if enable_pose_drawing:
                            frame = draw_poses(frame, results, self.names, line_thickness=self.opt.line_thickness,
                                               hide_conf=self.opt.hide_conf, hide_labels=self.opt.hide_labels)

                        # pose classifier
                        #self.pose_classifier.symbolic_classify(results, 10)

            # send_to_stream(frame)
            # Send to the shared database (for internal app component)
            # Send the stream video

            output[i] = frame
            frame = None
        j += 1
        j %= len(self.cameras)
        return output

    def close_cameras(self):
        for i, cam in enumerate(self.cameras):
            cam.stop()
