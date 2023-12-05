import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Camera():

    def __init__(self, X01_prime, Y01_prime, thetaCam, f, i_width, i_height):
        self.f=2.8*10**(-3)   # focal length in meters
        self.lbda = 1  # inverse depth of a 3D point
        self.Xcam = np.array([[X01_prime],
                              [Y01_prime]])  # coordonnées de la camera dans RO
        self.thetaCam = thetaCam    # orientation of the camera

        self.c=np.array([i_width/2,i_height/2])  # coordinate of the camera center compared to image referential
        # question de l'unité: en pixel ou en m?
        self.K=np.array([[self.f, 0, self.c[0]],
                         [0, self.f, self.c[1]],
                         [0, 0, 1]])            # intrasinc camera matrix
        self.R=np.array([[np.cos(self.thetaCam), -np.sin(self.thetaCam), 0],
                         [np.sin(self.thetaCam), np.cos(self.thetaCam), 0],
                         [0, 0, 1]])  # rotation matrix, rotation according to Z
        self.t=np.array([[X01_prime],
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
            X=np.linalg.inv(self.P)@(self.lbda*x)
            return X
        except np.linalg.LinAlgError:
            print("Singular Matrix, computed pseudo inverse")
            return np.linalg.pinv(self.P)@(self.lbda*x)


if __name__=='__main__':
    myCam=Camera(2,9,45,0.5)   # test camera

    #Y=np.array([1, 5, 6, 3/2,])    # yolo vector
    #x=Y[1:3]# pixels coordinates
    x=np.array([[2],[15]])
    X=myCam.pixelToWorld(x) # mapping to world coordinates

    print(X)


