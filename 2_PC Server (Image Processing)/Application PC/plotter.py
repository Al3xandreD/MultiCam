import sys
import numpy as np
import threading
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from pyqtgraph import PlotWidget
import pyqtgraph as pg


class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Plot")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create a pyqtgraph PlotWidget
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        # Initialize data for plotting
        self.x_data = np.linspace(0, 10, 100)
        self.y_data = np.zeros_like(self.x_data)

        # Plot data
        self.plot = self.plot_widget.plot(self.x_data, self.y_data, pen='r')

        self.is_plotting = False
        self.plot_thread = None

    def start_plotting(self):
        self.is_plotting = True
        self.plot_thread = threading.Thread(target=self.update_plot)
        self.plot_thread.start()

    def stop_plotting(self):
        self.is_plotting = False

    def update_plot(self):
        while self.is_plotting:
            start_time = time.time()
            y = np.sin(self.x_data + start_time)
            self.y_data[:-1] = self.y_data[1:]
            self.y_data[-1] = y[-1]
            self.plot.setData(self.x_data, self.y_data)
            # Adjust the delay to control the plot update rate
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.1 - elapsed_time))