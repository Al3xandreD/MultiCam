import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Camera():

    def __init__(self, X01_prime, Y01_prime, thetaX, thetaY, thetaZ, i_width, i_height):
        self.f=2.8*10**(-3)   # focal length in meters
        self.lbda = 1  # inverse depth of a 3D point
        self.Xcam = np.array([[X01_prime],
                              [Y01_prime]])  # coordonnées de la camera dans RO
        self.thetaX = thetaX
        self.thetaY = thetaY
        self.thetaZ = thetaZ    # orientation of the camera

        self.c=np.array([i_width/2,i_height/2])  # coordinate of the camera center compared to image referential

        self.K=np.array([[self.f, 0, self.c[0]],
                         [0, self.f, self.c[1]],
                         [0, 0, 1]])  # intrinsic camera matrix

        rZ=np.array([[np.cos(self.thetaZ), -np.sin(self.thetaZ), 0],
                    [np.sin(self.thetaZ), np.cos(self.thetaZ), 0],
                    [0, 0, 1]])  # rotation matrix, rotation according to Z
        rX=np.array([[1, 0, 0],
                    [0, np.cos(self.thetaX), -np.sin(self.thetaX)],
                    [0, np.sin(self.thetaX), np.cos(self.thetaX)]])

        rY=np.array([[np.cos(self.thetaY), 0, -np.sin(self.thetaY)],
                    [0, 1, 0],
                    [np.sin(self.thetaY), 0, np.cos(self.thetaY)]])
        self.R=rX@rY@rZ # rotation matrix

        self.t=-self.R@np.array([[X01_prime],
                                 [Y01_prime],
                                 [0]])  # translation vector

        self.P = self.K @ np.hstack((self.R,self.t))
        self.P=np.vstack((self.P, np.array([0, 0, 0, 1])))  # homogenous coordinates, allows to invert

    def setOpticC(self, i_width, i_height):
        """
        Sets the optic center, assuming it is the image center
        :param i_width: image width
        :param i_height: image center
        :return: None
        """
        c=np.array([i_width/2,i_height/2])

    def pixelToWorld(self, x):
        """
        Projects pixels coordinates into world coordinates
        :param x: pixel coordinates of a 3D point
        returns: X world coordinates of the same point
        """
        try:
            if x.shape==(2,1):
                x=np.vstack((x,np.array([[0], [1]]))) # in homogenous coordinates
            if x.shape==(3,1):
                x=np.vstack((x, np.array([[1]])))   # homogenous coordinates
            X=np.linalg.inv(self.P)@(self.lbda*x)
            return X
        except np.linalg.LinAlgError:
            print("Singular Matrix, computed pseudo inverse")
            return np.linalg.pinv(self.P)@(self.lbda*x)


if __name__=='__main__':
    theta=48*np.pi/180
    myCam=Camera(0.0,0.0,0, theta, 0,0.8, 1.06)   # test camera

    #Y=np.array([1, 5, 6, 3/2,])    # yolo vector
    #x=Y[1:3]# pixels coordinates
    x=np.array([[0.379],[0.549]])
    X=myCam.pixelToWorld(x) # mapping to world coordinates

    print(X)


