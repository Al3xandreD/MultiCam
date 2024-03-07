# Ref : https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
from threading import Thread
from os.path import dirname, abspath
import os
from network.localnetwork import*

import cv2
import imutils
import time
import argparse
import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QStyle, QGroupBox, QComboBox, QScrollArea
import pyshine as ps
from ui.container import UiContainer
from ui.auto_sizeable_qwidget import AutoSizeQGroupBox, AutoSizeQWidget
from ui.radio_buttons import RadioButtons
from ui.cameraview import CameraView

from core_app import Main


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam')  # video source
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
        self.opt = parse_opt()

        # Define common styles
        self.common_border_style = "border-radius: 4px; border:1px solid rgb(250, 250, 250);"

        # UI construction begin here
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Define the main grid layout of the app
        self.root_gridlayout = QtWidgets.QGridLayout(self.centralwidget)
        self.root_gridlayout.setObjectName("root_gridlayout")

        # Left side menu container
        self.leftside_menu_container = QGroupBox()
        self.root_gridlayout.addWidget(self.leftside_menu_container, 0, 0, 1, 1)
        self.leftside_menu_container.setFixedSize(250, 900)
        self.leftside_menu_container.setObjectName("leftside_menu_container")
        self.leftside_menu_container.setStyleSheet(
            "QGroupBox#" + self.leftside_menu_container.objectName() + "{border-radius: 4px; border:1px solid rgb(200, 200, 200);}")

        # Left side menu restrictor
        self.leftside_menu_restrictor = QGroupBox(self.leftside_menu_container)
        self.leftside_menu_restrictor.setObjectName("leftside_menu_restrictor")
        self.leftside_menu_restrictor.setStyleSheet(
            "QGroupBox#" + self.leftside_menu_restrictor.objectName() + "{border-radius: 4px; border:0 solid rgb(0, 200, 200);}")
        self.leftside_menu_restrictor.setFixedSize(200, 400)

        # Left side menu - Grid
        self.leftside_menu_grid = QtWidgets.QGridLayout(self.leftside_menu_restrictor)
        self.leftside_menu_grid.setColumnMinimumWidth(0, 50)
        self.leftside_menu_grid.setRowMinimumHeight(0, 50)

        # Left side menu - Grid - Network settings
        self.network_settings_container = AutoSizeQGroupBox()
        self.leftside_menu_grid.addWidget(self.network_settings_container, 0, 0, 1, 1)
        self.network_settings_container.setObjectName("network_settings_container")
        self.network_settings_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.network_settings_container.setStyleSheet(f"QGroupBox#{self.network_settings_container.objectName():s}" + "{" + self.common_border_style + "}")

        # Left side menu - Grid - Network settings - current wifi network ssid
        _, lan_ssid = get_current_lan_ssid()
        self.network_settings_ssid = QtWidgets.QLabel()
        self.network_settings_ssid.setText(f"SSID : {lan_ssid:s}")
        self.network_settings_ssid.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.network_settings_container.addItem(self.network_settings_ssid)
        self.network_settings_ssid.setObjectName("network_settings_ssid")
        self.network_settings_ssid.setStyleSheet(
            f"QGroupBox#{self.network_settings_ssid.objectName():s}" + "{" + self.common_border_style + "}")
        self.network_settings_ssid.setFixedSize(self.network_settings_ssid.width(), 30)
        self.network_settings_ssid.setMaximumSize(190, 90)

        # Left side menu - Grid - Cameras settings
        self.cameras_settings_container = AutoSizeQGroupBox()
        self.leftside_menu_grid.addWidget(self.cameras_settings_container, 1, 0, 1, 1)
        self.cameras_settings_container.setObjectName("cameras_settings_container")
        # self.cameras_settings_container.setStyleSheet(f"QGroupBox#{self.cameras_settings_container.objectName():s}" + "{" + self.common_border_style + "}")

        # Left side menu - Grid - Cameras settings - source
        self.cameras_settings_source_cbox = QtWidgets.QComboBox()
        self.cameras_settings_source_cbox.addItem("Webcam")
        self.cameras_settings_source_cbox.addItem("Multicam")
        self.cameras_settings_source_cbox.sizeAdjustPolicy()
        self.cameras_settings_source_cbox.currentIndexChanged.connect(self.set_source)
        self.cameras_settings_container.addItem(self.cameras_settings_source_cbox)

        # Left side menu - Grid - Cameras settings - cameras count
        self.cameras_settings_count_label = QtWidgets.QLabel()
        self.cameras_settings_count_label.setText("Connected : 1 / 5")
        self.cameras_settings_count_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                        QtWidgets.QSizePolicy.Expanding)
        # TODO self.setting_sources_cbox.setEnabled(False)

        # Left side menu - Grid - Detection settings
        self.detection_settings_container = AutoSizeQGroupBox()
        self.leftside_menu_grid.addWidget(self.detection_settings_container, 2, 0, 1, 1)
        self.detection_settings_container.setObjectName("detection_settings_container")
        # self.detection_settings_container.setStyleSheet(f"QGroupBox#{self.detection_settings_container.objectName():s}" + "{" + "self.common_border_style" + "}")

        # Left side menu - Grid - Detection settings - CPU-GPU Radio Buttons
        self.cpu_gpu_radiobtns = RadioButtons(("CPU", "GPU"), "h")
        self.cpu_gpu_radiobtns.setTitle("Run on: ")
        self.detection_settings_container.addItem(self.cpu_gpu_radiobtns)
        self.detection_settings_container.updateSizeWrapping(QSize(50, 20))
        self.cpu_gpu_radiobtns.toggled.connect(self.set_computing_device)
        self.cpu_gpu_radiobtns.setCurrentIndex(1)

        # Left side menu - Grid - Detection settings - bbox line thickness
        self.detection_bbox_thickness_label = QtWidgets.QLabel()
        self.detection_bbox_thickness_label.setText("Bounding box line thickness ")
        self.detection_settings_container.addItem(self.detection_bbox_thickness_label)
        self.detection_bbox_thickness_hslider = QtWidgets.QSlider()
        self.detection_bbox_thickness_hslider.setOrientation(QtCore.Qt.Horizontal)
        self.detection_bbox_thickness_hslider.setMaximumSize(50, 25)
        self.detection_settings_container.addItem(self.detection_bbox_thickness_hslider)

        # Cameras Start/Stop.
        self.camera_controls_container = QtWidgets.QHBoxLayout()
        self.camera_controls_container.setObjectName("camera_controls_container")
        self.root_gridlayout.addLayout(self.camera_controls_container, 1, 0, 1, 1)
        self.startnstop_btn = QtWidgets.QPushButton()
        self.startnstop_btn.setObjectName("startnstop_btn")
        self.camera_controls_container.addWidget(self.startnstop_btn)

        # Main (right side) : tab widget
        self.main_tab_root = QtWidgets.QTabWidget()
        self.root_gridlayout.addWidget(self.main_tab_root, 0, 2, 2, 1)
        self.main_tab_root.setMinimumSize(500, 500)
        self.cameras_tab = QWidget()
        self.map_tab = QWidget()
        self.specific_view_tab = QWidget()
        self.main_tab_root.addTab(self.cameras_tab, "Cameras")
        self.main_tab_root.addTab(self.map_tab, "Floor plans")
        self.main_tab_root.addTab(self.specific_view_tab, "Specific view")

        # Main (right side) - Camera tab
        self.scrollAreaContainer = QtWidgets.QHBoxLayout(self.cameras_tab)  # To make content expand
        self.scrollArea = QScrollArea()
        self.scrollAreaContainer.addWidget(self.scrollArea)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.viewgrid = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.viewgrid.setRowMinimumHeight(0, 250)
        self.viewgrid.setColumnMinimumWidth(0, 250)
        self.viewgrid.setObjectName("gridLayout")

        # Main (right side) - Specific Camera View tab
        self.specific_view_hb = QtWidgets.QVBoxLayout(self.specific_view_tab)
        self.specific_cam_view = CameraView()
        self.specific_view_hb.addWidget(self.specific_cam_view)
        self.specific_view_id = 0

        spacerItem = QtWidgets.QSpacerItem(150, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.root_gridlayout.addItem(spacerItem, 1, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)

        # TODO Manage camera toggle
        self.startnstop_btn.clicked.connect(self.toggle_run)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Added code here
        self.filename = 'Snapshot  ' + str \
            (time.strftime("%Y-%b-%d at %H.%M.%S %p")) + '.png'  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.brightness_value_now = 0  # Updated brightness value
        self.blur_value_now = 0  # Updated blur value
        self.fps = 0
        self.started = False

    def set_source(self):
        self.opt.source = f"{self.cameras_settings_source_cbox.currentIndex():d}"
        print(f"Source choice: {self.opt.source:s}")

    def set_computing_device(self):
        choice = self.cpu_gpu_radiobtns.current_index
        print(f'device choice: {choice:d}')
        self.opt.device = "0" if choice else "cpu"

    def toggle_run(self):
        if self.started:
            self.started = False
            self.onStop()
        else:
            self.started = True
            self.onStart()

    def onStart(self):
        self.startnstop_btn.setText('Stop')
        self.enableRuntimeGUI(True)
        self.enableNoRuntimeGUI(False)
        self.run()

    def onStop(self):
        self.startnstop_btn.setText('Start')
        self.enableRuntimeGUI(False)
        self.enableNoRuntimeGUI(True)

        # Destroy all multi cam views content and clean the specific view
        self.specific_cam_view.setPixmap(QPixmap())
        self.clearLayout(self.viewgrid)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def run(self):
        """ This function will load the camera devices, obtain images
            and update and draw them on camera views
        """
        ############# Get options from GUI Start Menu ################

        #############################################################
        self.main_app = Main(self.opt)
        self.network_settings_ssid.setText(f"SSID : {self.main_app.lan_ssid:s}")
        self.network_settings_ssid.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.network_settings_container.updateSizeWrapping()

        cameras = self.main_app.cameras  # TODO Replace with a result output member of Main() class
        self.camviews = [None for i in range(len(cameras))]

        for i, cam in enumerate(cameras):
            self.camviews[i] = CameraView()
            self.camviews[i].setText("")
            self.camviews[i].setObjectName(f"label{i:d}")
            self.camviews[i].setMaximumSize(1000, 1000)
            self.camviews[i].doubleclicked.connect(lambda val=i: self.open_specific_view(val))
            cols_count = 3  # TODO set it as a constant above
            self.viewgrid.addWidget(self.camviews[i], i // cols_count, i % cols_count, 1, 1)

        self.specific_cam_view.setMaximumSize(1000, 1000)

        while True:
            QtWidgets.QApplication.processEvents()
            self.images = self.main_app.update()
            self.update()

            key = cv2.waitKey(1) & 0xFF
            if not self.started:
                print('Loop break')
                break
        self.main_app.close_cameras()

    def open_specific_view(self, index):
        self.main_tab_root.setCurrentIndex(2)
        self.specific_view_id = index
        print(f'"opened tab no : {index:d}')

    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """

        if (self.camviews[0] != None):
            adapted_width = self.camviews[0].width()

        # Update multiple cameras views
        if self.main_tab_root.currentIndex() == 0:
            for i, image in enumerate(self.images):
                frame = imutils.resize(image, width=adapted_width)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qframe = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
                self.camviews[i].setPixmap(QPixmap.fromImage(qframe))

        # Update specific view
        if self.main_tab_root.currentIndex() == 2:
            frame = imutils.resize(self.images[self.specific_view_id], width=self.specific_cam_view.width(),
                                   height=self.specific_cam_view.height())
            qframe = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
            self.specific_cam_view.setPixmap(QtGui.QPixmap.fromImage(qframe))

        # # Here we add display text to the image
        # text = 'FPS: ' + str(self.fps)
        # img = ps.putBText(img, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0,
        #                   background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MultiCam"))
        self.startnstop_btn.setText(_translate("MainWindow", "Start"))
        self.network_settings_container.setTitle(_translate("MainWindow", "Network : ", None))
        self.cameras_settings_container.setTitle(_translate("MainWindow", "Cameras : ", None))
        self.detection_settings_container.setTitle(_translate("MainWindow", "Detection : ", None))

    # Put on this method all GUI that need to be Disabled when start button is activated
    def enableNoRuntimeGUI(self, state: bool):
        self.cameras_settings_source_cbox.setEnabled(state)
        self.cpu_gpu_radiobtns.setEnabled(state)

    # Put on this method all GUI that need to be Activated when start button is activated
    def enableRuntimeGUI(self, state: bool):
        print()


if __name__ == "__main__":
    print(1)
    app = QtWidgets.QApplication(sys.argv)
    print(2)
    MainWindow = QtWidgets.QMainWindow()
    print(3)
    ui = Ui_MainWindow()
    print(4)
    ui.setupUi(MainWindow)
    print(5)
    MainWindow.show()
    print(6)
    # TODO close properly the core_app
    # For some reasons camera still opened after closing the window app
    sys.exit(app.exec_())


