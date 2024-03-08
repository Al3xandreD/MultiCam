import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget, QStyle, QGroupBox, QComboBox, QSizePolicy
import pyshine as ps

"""With PyQt5 QWidget are not auto resizeable (thanks to size hint) but QVBoxLayout and QHBoxLayout are,
# but it is not possible to add a non QWidget to a grid. This class permit to combine the advantages of both
# QWidget (place on grid) and QVBoxLayout (wrap content)."""


class AutoSizeQWidget(QWidget):

    def __init__(self, orientation='v'):
        super().__init__()
        self.orientation = orientation
        self.size_offset = QSize(0, 15)
        if (orientation == 'h'):
            self.box = QtWidgets.QHBoxLayout(self)
        else:
            self.box = QtWidgets.QVBoxLayout(self)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def addItem(self, other):

        if isinstance(other, QWidget):
            self.box.addWidget(other)
        elif isinstance(other, QtWidgets.QLayout):
            self.box.addLayout(other)
        else:
            self.box.addItem(other)

        # Wrap content (set automatically size):
        self.updateSizeWrapping()

    def setSizeOffset(self, size_offset: QSize):
        self.size_offset = size_offset

    def updateSizeWrapping(self, size_offset: QSize = QSize(0, 0)):
        self.setFixedSize(self.box.totalSizeHint() + self.size_offset)



class AutoSizeQGroupBox(QGroupBox):

    def __init__(self, orientation='v'):
        super().__init__()
        self.orientation = orientation
        self.size_offset = QSize(0, 15)
        if (orientation == 'h'):
            self.box = QtWidgets.QHBoxLayout(self)
        else:
            self.box = QtWidgets.QVBoxLayout(self)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def addItem(self, other):

        if isinstance(other, QWidget):
            self.box.addWidget(other)
        elif isinstance(other, QtWidgets.QLayout):
            self.box.addLayout(other)
        else:
            self.box.addItem(other)

        # Wrap content (set automatically size):
        self.updateSizeWrapping()

    def setSizeOffset(self, size_offset: QSize):
        self.size_offset = size_offset

    def updateSizeWrapping(self, size_offset: QSize = QSize(0, 0)):
        self.setFixedSize(self.box.totalSizeHint() + self.size_offset)

    def render_(self):
        self.box.setParent(self)
