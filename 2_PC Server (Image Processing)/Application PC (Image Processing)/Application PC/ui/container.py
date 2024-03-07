import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget, QStyle, QGroupBox, QComboBox
import pyshine as ps


class UiContainer(QGroupBox):
    def __init__(self, border_width=1, border_radius=1, border_color="solid rgb(50, 50, 50)"):
        super().__init__()
        self.offset = -2
        self.frame = QtWidgets.QWidget(self)
        self.frame.setFixedSize(self.size().width() + self.offset, self.size().height() + self.offset)
        self.frame.setStyleSheet(f"border-radius: {border_radius:d}px; border:{border_width:d}px {border_color:s};")

    def setMaximumSize(self, maxw: int, maxh: int):
        super().setMaximumSize(maxw, maxh)
        self.frame.setFixedSize(self.size().width() + self.offset, self.size().height() + self.offset)
        self.__auto_center_frame()

    def setFixedSize(self, size: QSize):
        super().setFixedSize(size)
        self.frame.setFixedSize(self.size().width() + self.offset, self.size().height() + self.offset)
        self.__auto_center_frame()



    def setStyleSheet(self, st):
        super().setStyleSheet(self.styleSheet())
        self.frame.setStyleSheet(st)
        self.__auto_center_frame()

    def __auto_center_frame(self):
        self.frame.setGeometry(-self.offset // 2, -self.offset // 2, self.frame.width(), self.frame.height())
