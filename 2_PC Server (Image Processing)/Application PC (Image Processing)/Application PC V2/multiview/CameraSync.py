import numpy as np
import MultiviewHomogenous as mh
import cv2
import scipy
from codeRayan import UrlCamera


class Camera(UrlCamera):
    def __init__(self, status, l_points, source):

        # variable relative à l'initialisation des caméras
        super().__init__(source)
        self.l_points=l_points  # liste point cam1 pour 8points algorithm

        # variables relatives à la structure de la paire
        self.status=status

        # variables relatives aux paramètres de la camera
        self.e=np.zeros((3,1))  # epipole
        if self.status=="master":
            self.P=np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix, identity for master camera
        else:
            self.P=np.zeros((3,4))
