import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget, QStyle, QGroupBox, QComboBox, QLabel
import pyshine as ps


class CameraView(QLabel):
    doubleclicked = QtCore.pyqtSignal()

    def __init__(self, parent=None, view_id=0):
        super().__init__(parent)
    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.doubleclicked.emit()
