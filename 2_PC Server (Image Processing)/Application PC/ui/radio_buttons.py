import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget, QStyle, QGroupBox, QComboBox, QRadioButton
import pyshine as ps
from .auto_sizeable_qwidget import AutoSizeQWidget, AutoSizeQGroupBox

"""This class add multi-radio buttons group that is not initially present in PyQt5."""


class RadioButtons(AutoSizeQGroupBox):
    toggled = QtCore.pyqtSignal()

    def __init__(self, items_labels=(), orientation='v'):
        super().__init__(orientation=orientation)
        self.current_index = 0
        self.item_count = 0
        for item in items_labels:
            self.addItemBtn(item)

    def addItemBtn(self, item_label=""):
        radiobtn = QtWidgets.QRadioButton()
        radiobtn.setText(item_label)
        radiobtn.toggled.connect(lambda var=self.item_count: self.__setCurrentIndex(var))
        if self.item_count == 0:
            radiobtn.setChecked(True)
        self.addItem(radiobtn)
        self.item_count += 1


        # Wrap content (set automatically size):
        self.updateSizeWrapping()

    def __setCurrentIndex(self, index):
        if self.item_count > 0:
            if self.children()[index+1].isChecked():
                self.current_index = index
                self.toggled.emit()

    def setCurrentIndex(self, index):
        self.children()[index + 1].setChecked(1)