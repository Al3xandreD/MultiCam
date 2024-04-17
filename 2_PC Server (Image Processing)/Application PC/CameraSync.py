import numpy as np
from camera import GenericCamera


class Camera(GenericCamera):
    """
    Camera class extending GenericCamera to add support for epipolar geometry
    """
    def __init__(self, source, rot_angle, source_is_auto_refresh):

        # variable relative à l'initialisation des caméras
        super().__init__(source, rot_angle, source_is_auto_refresh)

        # variables relatives à la structure de la paire
        self.status=None

        self.e = np.zeros((3, 1))  # epipole

    def check_status(self):

        # variables relatives aux paramètres de la camera
        if self.status=="master":
            self.P=np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix, identity for master camera
            self.F=None
        else:
            self.P=np.zeros((3,4))
