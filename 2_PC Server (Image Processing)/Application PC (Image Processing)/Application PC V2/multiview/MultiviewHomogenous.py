import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image

def normalisation(l_points):
    '''
    Normalise the points before computing fundamental matrix
    :param l_points:
    :return:
    '''

    center=np.mean(l_points, axis=1)    # coordinates of center of points

    m_dist=0    # mean distance to the center
    for k in range(l_points.shape[1]):
        m_dist+=np.sqrt(np.sum(np.array([[l_points[0][k]],[l_points[1][k]]])-center)**2)
    m_dist=m_dist/l_points.shape[1]

    T=np.array([[1, 0, -center[0]], # translation matrix
                [0, 1, -center[1]],
                [0, 0, 1]])

    l_points=np.vstack((l_points, np.ones((1, l_points.shape[1])))) # working in homogenous coordinates
    l_translated=T@l_points # translated points

    scale_f = np.sqrt(2) / m_dist
    S=np.array([[scale_f, 0, 0],
                [0, scale_f, 0],
                [0, 0, 1]])
    l_scaled=S@l_translated # translated points

    return l_scaled, S@T

def computeAbis(l_points,l_points_prime):
    '''
    Compute the A matrix, necessary to compute the fundamental matrix, used for 8points algorithm
    :param l_points: array containing the coordinate of points on first camera
    :param l_points_prime: array containing the coordinate of points on second camera
    :return: A
    '''
    A = np.zeros((l_points.shape[1], 9))
    for k in range(l_points.shape[1]):   # for all points
        A[k] = np.array([l_points[0][k]*l_points_prime[0][k], l_points[0][k]*l_points_prime[1][k], l_points[0][k], l_points[1][k]*l_points_prime[0][k], l_points[1][k]*l_points_prime[1][k], l_points[1][k], l_points_prime[0][k], l_points_prime[1][k], 1])
    return A

def fundamentalMatrixSvd(A,T):
    '''
    Computes the fundamental matrix using singular value decomposition method
    :param A: 2D array of the fundamental
    :param T: 2D array for unnormalizing F
    :return:  fundamental matrix
    '''

    U,S,V=np.linalg.svd(A.T@A)

    # smallest singular value is the last one in S
    F=V[-1].reshape(3,3)    # retrieving the associated eigenvector

    # applying constraint on F rank 2 by zeroing out the last singular value
    U,S,V=np.linalg.svd(F)
    S[2]=0
    F=U@(np.diag(S)@V)

    return T.T@F@T  # un normalized F

def computeFirstEpipole(F):
    '''
    Computes the epipole of the first camera according to least-square method, Fe=0
    :param F: Fundamental matrix
    :return: epi, epipole
    '''

    U,S,V=np.linalg.svd(F)
    epi=V[-1]
    return (epi/epi[2]).reshape(3,1)

def computeSecondEpipole(F):
    '''
    Computes the epipole of the second camera according to least-square method, e'.T@F=0
    :param F: Fundamental matrix
    :return: e_prime, epipole
    '''

    return computeFirstEpipole(F.T)

def computePfromE(F,K, Kprime):
    '''
    Computes the second camera matrix, supposing intrinsic matrix is known
    :param F: Fundamental matrix
    :param K:
    :param Kprime:
    :return: P camera matrix
    '''

    w=np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    E=Kprime.T@F@K
    U,S,V=np.linalg.svd(E)
    R1=U@w@V
    R2=U@w.T@V

    T1=U[:,2]
    T2=-U[:,2]

    return None
def computePfromF(F,e,e_prime):
    '''
    Computes the second camera matrix if uncalibrated
    :param F: fundamental matrix
    :param e: first epipole
    :param e_prime: second epipole
    :return: camera matrix
    '''
    skew_e=np.array([[0, -e[2][0], e[1][0]],
                     [e[2][0], 0, -e[0][0]],
                     [-e[1][0], e[0][0], 0]])   # skew matrix associated with the epipole
    Pprime=np.hstack((skew_e@F,e_prime)) #TODO doute sur skew_e
    return Pprime


def point_triangulation(point_x, point_xprime, P, Pprime):
    '''
    Triangulates a point from two images points via svd
    :param point_x: first point on first camera
    :param point_xprime: second point on second camera
    :param P: first camera matrix
    :param Pprime: second camera matrix
    :return: X coordinates of the triangulated point
    '''

    # building A

    A1=np.array([point_x[1][0]*P[2].T-P[1].T,
                 P[0].T-point_x[0][0]*P[2].T])

    A2=np.array([point_xprime[1][0]*Pprime[2].T-Pprime[1].T,
                 Pprime[0].T-point_xprime[0][0]*Pprime[2].T])

    A=np.vstack((A1,A2))


    U,S,V=np.linalg.svd(A)
    X=V[-1:4]
    return X

def computeLine(F,point, prime=False):
    '''
    Computes the epipolar line passing by a point
    :param F: Associated fundamental matrix
    :param point:
    :param prime: boolean to indicate if used for the second camera or not
    :return:
    '''
    if prime==True:
        return F@(np.vstack((point,np.array([1])))) # second camera

    else:
        return (F.T @ (np.vstack((point_prime, np.array([1])))))  # first camera


def plotImages(path1):
    '''
    plot the images, epipoles, epipolar lines and interest points
    :param path1: path of image
    :return: None
    '''
    try:
        img = np.asarray(Image.open(path1))
    except TypeError:
        print("Error while opening image")
    else:
        fig = plt.figure()
        plt.imshow(img)

def plotEpipoles(e):
    '''
    Plot epipoles
    :param e:
    :return: None
    '''
    plt.scatter(e[0][0], e[1][0], c="green", label='epipole')

def plotEpipolarLines(l, e, point):
    '''
    Plots epipolar lines
    :param l:
    :return: None
    '''
    endpoint = l + e
    plt.plot([point[0][0], endpoint[0][0]], [point[1][0], endpoint[1][0]], label='epipolar line')

def plotInterestPoint(point):
    '''
    Plots interests points
    :param point:
    :return: None
    '''
    plt.scatter(point[0][0], point[1][0], c="red", label='point of interest')



if __name__=='__main__':

    # Merton dataset
    "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/"
    p1 = "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampaA.png"
    p2 = "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampaB.png"

    # images correspondances
    # treating extraction from ground truth in matlab file
    mat=scipy.io.loadmat("/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/kampa_vpts.mat")
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

    l_normalized,T=normalisation(l_points)  # normalizing points
    l_normalized_prime,T=normalisation(l_points_prime)

    A=computeAbis(l_points, l_points_prime) # matrix for computing F
    F=fundamentalMatrixSvd(A,T) # fundamental matrix
    e=computeFirstEpipole(F)    # epipole
    e_prime=computeSecondEpipole(F)
    P=np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix of the first camera, identity
    Pprime=computePfromF(F, e, e_prime)

    # points of interest
    point1, point2 = np.array([[1.528833000000000e+02], [2.243500000000000e+02]]), np.array([[1.104979000000000e+02], [1.632479000000000e+02]])

    # triangulation
    X=point_triangulation(point1, point2, P, Pprime)
    print('This is X')
    print(X)

    # plots

    # Image from first camera
    plotImages(p1)
    plotEpipoles(e)
    for k in range(l_points_prime.shape[1]):
        point = np.array([[l_points[0][k]], # same dimensions for l_point_prime and l_points
                          [l_points[1][k]]])
        point_prime=np.array([[l_points_prime[0][k]],
                        [l_points_prime[1][k]]])

        l = computeLine(F, point_prime)   #

        plotInterestPoint(point)
        plotEpipolarLines(l, e, point)
    #plt.legend()

    # Image from second camera
    plotImages(p2)
    plotEpipoles(e_prime)
    for k in range(l_points.shape[1]):
        point = np.array([[l_points[0][k]],
                          [l_points[1][k]]])
        point_prime = np.array([[l_points_prime[0][k]],
                                [l_points_prime[1][k]]])

        l_prime = computeLine(F, point, prime=True)

        plotInterestPoint(point_prime)
        plotEpipolarLines(l_prime, e_prime, point_prime)

    #plt.legend()

    plt.show()