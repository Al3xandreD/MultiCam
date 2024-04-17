# Ref : https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
from threading import Thread
from os.path import dirname, abspath
import os
import cv2
import imutils
import time
import argparse
import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget
import pyshine as ps

from core_app import Main

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='video0.mp4', help='video/0 for webcam')  # video source
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


opt = parse_opt()
main_app = Main(opt)
cameras = main_app.cameras
n = len(cameras)

while True:
    images = main_app.update()

    for i, frame in enumerate(images):
        frame = imutils.resize(frame, width=800)
        cv2.imshow(f'Live cam no : {i + 1:d}/{n:d}', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()



