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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(313, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.pushButton_2.clicked.connect(self.run)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Added code here
        self.filename = 'Snapshot  ' + str \
            (time.strftime("%Y-%b-%d at %H.%M.%S %p")) + '.png'  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.brightness_value_now = 0  # Updated brightness value
        self.blur_value_now = 0  # Updated blur value
        self.fps = 0
        self.started = False

    def run(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        if self.started:
            self.started = False
            self.pushButton_2.setText('Start')
        else:
            self.started = True
            self.pushButton_2.setText('Stop')

        main_app = Main(opt)
        cameras = main_app.cameras
        self.camviews = [None for i in range(len(cameras))]
        self.viewgrid = QtWidgets.QGridLayout()
        self.viewgrid.setObjectName("gridLayout")
        for i, cam in enumerate(cameras):
            self.camviews[i] = QtWidgets.QLabel(self.centralwidget)
            self.camviews[i].setText("")


            self.camviews[i].setObjectName(f"label{i:d}")

            self.camviews[i].setMaximumSize(400, 400)
            cols_num = 3
            self.viewgrid.addWidget(self.camviews[i], i // cols_num, i % cols_num, 1, 1)

        self.horizontalLayout.addLayout(self.viewgrid)

        while True:
            QtWidgets.QApplication.processEvents()
            self.images = main_app.update()
            self.update()
            key = cv2.waitKey(1) & 0xFF
            if self.started == False:
                break
                print('Loop break')



    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        for i, frame in enumerate(self.images):
            frame = imutils.resize(frame, width=400)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qframe = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.camviews[i].setPixmap(QtGui.QPixmap.fromImage(qframe))

        # # Here we add display text to the image
        # text = 'FPS: ' + str(self.fps)
        # img = ps.putBText(img, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0,
        #                   background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MultiCam"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        self.pushButton.setText(_translate("MainWindow", "Take picture"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
