import MultiviewHomogenous as mh
import numpy as np
import scipy


class BiCamera(object):
    """
    Associates a slave and a master camera, used for configuration.
    """
    def __init__(self, cam1, cam2):

        self.camMaster = cam1
        self.camSlave=cam2

    def init(self):
        '''
        Starts the camera handler and calls appropriates methods
        :return:
        '''

        left_image, right_image=self.captureChekerCalib()
        self.configuration(left_image, right_image)

    def captureChekerCalib(self):
        '''
        Acquires two photographs used for calibrating the cameras
        :return: left photograph and right photograph
        '''

        self.camMaster.start()
        self.camMaster.update(self.url)
        self.camSlave.start()
        self.camSlave.update(self.url)
        _, left_image=self.camMaster.read()
        _, right_image=self.camSlave.read()

        self.camMaster.stop()
        self.camSlave.stop()

        return left_image, right_image

    def configuration(self, left_image, right_image):
        '''
        Cameras configuration according to epipolar geometry.
        :return: None
        '''

        mtx1, dist1 = mh.calibrate_camera(left_image)
        mtx2, dist2 = mh.calibrate_camera(right_image)

        R, T, F = mh.stereo_calibrate(mtx1, dist1, mtx2, dist2, left_image, right_image)

        self.camMaster.e = mh.computeFirstEpipole(F)  # epipole
        self.camSlave.e = mh.computeSecondEpipole(F)
        self.camMaster.F=F
        self.camSlave.P = mh.computePfromF(F, self.camMaster.e, self.camSlave.e)

    def start(self):
        '''
        Starts the acquisition
        :return:
        '''

        self.camMaster.start()
        self.camSlave.start()

    def update(self):
        '''
        update the frames
        :return:
        '''
        self.camMaster.update(self.url)
        self.camSlave.update(self.url)
        _, left_image = self.camMaster.read()
        _, right_image = self.camSlave.read()

        return left_image, right_image

    def localisation(self, point_x, point_x_prime):
        '''

        :param point_x:
        :param point_x_prime:
        :return:
        '''

        for x, xprime in zip(point_x, point_x_prime):
            X=mh.point_triangulation(x, xprime, self.camMaster.P, self.camSlave.P)





if __name__ == '__main__':

    # Merton dataset
    p1 = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/MultiCam/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampaA.png"
    p2 = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/MultiCam/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampaB.png"

    # images correspondances
    # treating extraction from ground truth in matlab file
    mat=scipy.io.loadmat("/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/MultiCam/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampa_vpts.mat")
    # returns a dictionnary with variable names as keys and loaded matrices as values

    liste_values=list(mat.values())
    for k in range(3):
        liste_values.pop(0)

    tab=liste_values[0][0][0][1]    # array containing the point correspondences in homogenous coordinates

    l_points=np.array([[tab[0][0]],[tab[1][0]]])
    l_points_prime=np.array([[tab[3][0]],[tab[4][0]]])
    for k in range(1,len(tab[0])):
        point=np.array([[tab[0][k]],
                        [tab[1][k]]])
        point_prime=np.array([[tab[3][k]],
                              [tab[4][k]]])

        l_points=np.hstack((l_points, point))   # coordinates of correspondence points in camera 1
        l_points_prime=np.hstack((l_points_prime, point_prime))     # coordinates of correspondence points in camera 2

    # configuration
    source1=0
    source2=0
    C1=Camera("master", l_points, source1)
    C2=Camera("slave", l_points_prime, source2)

    myHandler=HandlerCamera(C1,C2)
    myHandler.configuration()

    # acquisition
    myHandler.camMaster.start()
    myHandler.camMaster.update(source1)
    myHandler.camSlave.start()
    myHandler.camSlave.update(source2)

    # localisation
    point1, point2 = np.array([[1.528833000000000e+02], [2.243500000000000e+02]]), np.array([[1.104979000000000e+02], [1.632479000000000e+02]])
    X=mh.point_triangulation(point1, point2, myHandler.camMaster.P, myHandler.camSlave.P)
    print('This is X')
    print(X)

    # plotting