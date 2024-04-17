import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
import cv2

def normalisation(l_points):
    '''
    Normalise the points before computing fundamental matrix
    :param l_points:
    :return:
    '''

    mean_x=np.mean(l_points[0])
    mean_y=np.mean(l_points[1])

    std_dev_x=np.std(l_points[0])
    std_dev_y=np.std(l_points[1])

    scale_x=np.sqrt(2)/std_dev_x
    scale_y=np.sqrt(2)/std_dev_y

    S=np.array([[scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]]) # scaling matrix

    T=np.array([[1, 0, -mean_x],
                [0, 1, -mean_y],
                [0, 0, 1]]) # translation matrix

    l_points=np.vstack((l_points, np.ones((1, l_points.shape[1]))))

    for k in range(l_points.shape[1]):
        l_points[:,k]=S@T@l_points[:,k]

    return l_points, S@T

    # center=np.mean(l_points, axis=1)    # coordinates of center of points
    #
    # m_dist=0    # mean distance to the center
    # for k in range(l_points.shape[1]):
    #     m_dist+=np.sqrt(np.sum(np.array([[l_points[0][k]],[l_points[1][k]]])-center)**2)
    # m_dist=m_dist/l_points.shape[1]
    #
    # T=np.array([[1, 0, -center[0]], # translation matrix
    #             [0, 1, -center[1]],
    #             [0, 0, 1]])
    #
    # l_points=np.vstack((l_points, np.ones((1, l_points.shape[1])))) # working in homogenous coordinates
    # l_translated=T@l_points # translated points
    #
    # scale_f = np.sqrt(2) / m_dist
    # S=np.array([[scale_f, 0, 0],
    #             [0, scale_f, 0],
    #             [0, 0, 1]])
    # l_scaled=S@l_translated # translated points
    #
    # return l_scaled, S@T

    # return normalized_point, T_x, T_y

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

def fundamentalMatrixNotNormalized(A):


    U,S,V = np.linalg.svd(A)
    # smallest singular value is the last one in S
    F=V[-1].reshape(3,3)    # retrieving the associated eigenvector
    # applying constraint on F rank 2 by zeroing out the last singular value
    U,S,V=np.linalg.svd(F)
    S[-1]=0
    F=U@(np.diag(S)@V)

    return F

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
    Pprime=np.hstack((skew_e@F,e_prime))
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

    A1=np.array([point_x[1][0]*P[2,:].T-P[1,:].T,
                 P[0,:].T-point_x[0][0]*P[2,:].T])

    A2=np.array([point_xprime[1][0]*Pprime[2,:].T-Pprime[1,:].T,
                 Pprime[0,:].T-point_xprime[0][0]*Pprime[2,:].T])

    A=np.vstack((A1,A2))


    U,S,V=np.linalg.svd(A.T@A)
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


def calibrate_camera(im):
    im = cv2.imread(im)
    images = [im]
    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 7  # number of checkerboard rows.
    columns = 7  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = im[0].shape[1]
    height = im[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.imshow('img', frame)
            cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)


    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, im_left, im_right):

    im_l = cv2.imread(im_left)

    im_r = cv2.imread(im_right)

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 7  # number of checkerboard rows.
    columns = 7  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = im_l.shape[1]
    height = im_l.shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space


    gray1 = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
    c_ret1, corners1 = cv2.findChessboardCorners(gray1, (7, 7), None)
    c_ret2, corners2 = cv2.findChessboardCorners(gray2, (7, 7), None)

    if c_ret1 == True and c_ret2 == True:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(im_l, (7, 7), corners1, c_ret1)
        cv2.imshow('img', im_l)

        cv2.drawChessboardCorners(im_r, (7, 7), corners2, c_ret2)
        cv2.imshow('img2', im_r)
        cv2.waitKey(500)

        objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,dist1,mtx2, dist2, (width, height), criteria=criteria,flags=stereocalibration_flags)

    return R, T, F


if __name__=='__main__':



    mtx1, dist1 = calibrate_camera('checkboard_left.jpg')
    mtx2, dist2 = calibrate_camera('checkboard_right.jpg')

    R, T,F = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'checkboard_left.jpg', 'checkboard_right.jpg')
    print("F opencv:", F)

    # test fonction maison sur checkerboard
    l_points_test=np.array([[1529.51, 1608.82, 1685.73, 1760.52, 1833.83, 1905.30, 1545.32, 1624.64, 1701.56, 1776.80],
                            [1319.47, 1308.50, 1297.98, 1287.27, 1277.40, 1267.05, 1402.52, 1390.60, 1378.54, 1367.31]])

    l_points_test_prime=np.array([[1706.79, 1763.75, 1821.97, 1880.17, 1940.00, 2000.06, 1695.07, 1752.01, 1810.10, 1867.99],
                                  [1374.03, 1382.33, 1390.96, 1400.04, 1408.85, 1418.03, 1436.99, 1446.02, 1455.02, 1464.10]])

    l_normalized_test,Transfo = normalisation(l_points_test)
    l_normalized_test_prime, _ = normalisation(l_points_test_prime)
    A_test=computeAbis(l_normalized_test, l_normalized_test_prime)
    F_test=fundamentalMatrixSvd(A_test,Transfo)
    print("F mano: ", F_test)

    A_not_normalized=computeAbis(l_points_test, l_points_test_prime)
    F_not_normalized=fundamentalMatrixNotNormalized(A_not_normalized)
    print("F not normalized: ", F_not_normalized)

    # les valeurs obtenues pour F sont différentes
    e_test=computeFirstEpipole(F)
    e_test_prime=computeSecondEpipole(F)
    P_test = np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix of the first camera, identity
    P_test_prime = computePfromF(F_test, e_test, e_test_prime)
    print("P' mano: ", P_test_prime)

    P_test_vs_opencv=computePfromF(F, e_test, e_test_prime)
    print("P' calculé par fonction maison et F issue d'opencv", P_test_vs_opencv)

    P_RT_prime=mtx2@np.concatenate([R, T], axis=-1)
    print("P' selon R,T", P_RT_prime)

    P_RT=mtx1@P_test

    point1_test=np.array([[1529.51], [1319.47]])
    point2_test=np.array([[1706.79], [1374.03]])

    X_test = point_triangulation(point1_test, point2_test, P_test, P_test_prime)
    X_test=X_test/X_test[0][-1]
    print("X pour triangulation mano: ",X_test)

    X_opencv=point_triangulation(point1_test, point2_test, P_RT, P_RT_prime)
    print("X triangulation opencv: ",X_opencv)

    