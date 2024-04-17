import numpy as np
from camera import GenericCamera


class CameraF(GenericCamera):
    """
    Camera class extending UrlCamera to add support for epipolar geometry
    """
    def __init__(self, status, source):

        # variable relative à l'initialisation des caméras
        super().__init__(source)

        # variables relatives à la structure de la paire
        self.status=status

        # variables relatives aux paramètres de la camera
        self.e=np.zeros((3,1))  # epipole
        if self.status=="master":
            self.P=np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix, identity for master camera
        else:
            self.P=np.zeros((3,4))
