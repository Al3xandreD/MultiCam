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

    # # Merton dataset
    # p1 = "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/castleA.png"
    # p2 = "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/castleB.png"
    #
    # # images correspondances
    # # treating extraction from ground truth in matlab file
    # mat = scipy.io.loadmat(
    #     "/Users/alexandredermouche/Desktop/MultiCam copy/PC Server (Image Processing)/Computer Vision R&D/Test dataset/castle_vpts.mat")
    # # returns a dictionnary with variable names as keys and loaded matrices as values
    #
    # liste_values=list(mat.values())
    # for k in range(3):
    #     liste_values.pop(0)
    #
    # tab=liste_values[0][0][0][1]    # array containing the point correspondences in homogenous coordinates
    #
    # l_points=np.array([[tab[0][0]],[tab[1][0]]])
    # l_points_prime=np.array([[tab[3][0]],[tab[4][0]]])
    # for k in range(1,len(tab[0])):
    #     point=np.array([[tab[0][k]],
    #                     [tab[1][k]]])
    #     point_prime=np.array([[tab[3][k]],
    #                           [tab[4][k]]])
    #
    #     l_points=np.hstack((l_points, point))   # coordinates of correspondence points in camera 1
    #     l_points_prime=np.hstack((l_points_prime, point_prime))     # coordinates of correspondence points in camera 2
    #
    # l_normalized,T=normalisation(l_points)  # normalizing points
    # l_normalized_prime,T=normalisation(l_points_prime)
    #
    # # computing matrices
    # A=computeAbis(l_normalized, l_normalized_prime) # matrix for computing F
    # F=fundamentalMatrixSvd(A,T) # fundamental matrix
    # e=computeFirstEpipole(F)    # epipole
    # e_prime=computeSecondEpipole(F)
    # P=np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix of the first camera, identity
    # Pprime=computePfromF(F, e, e_prime)
    #
    # # points of interest
    # point1, point2 = np.array([[32.566000000000020], [4.023540000000000e+02]]), np.array([[17.844000000000020], [4.252840000000000e+02]])
    #
    # # triangulation
    # X=point_triangulation(point1, point2, P, Pprime)
    # print('This is X')
    # print(X/X[0][-1])
    #
    # # plots
    #
    # # Image from first camera
    # plotImages(p1)
    # plotEpipoles(e)
    # for k in range(l_points_prime.shape[1]):
    #     point = np.array([[l_points[0][k]], # same dimensions for l_point_prime and l_points
    #                       [l_points[1][k]]])
    #     point_prime=np.array([[l_points_prime[0][k]],
    #                     [l_points_prime[1][k]]])
    #
    #     l = computeLine(F, point_prime)
    #
    #     plotInterestPoint(point)
    #     plotEpipolarLines(l, e, point)
    # #plt.legend()
    #
    # # Image from second camera
    # plotImages(p2)
    # plotEpipoles(e_prime)
    # for k in range(l_points.shape[1]):
    #     point = np.array([[l_points[0][k]],
    #                       [l_points[1][k]]])
    #     point_prime = np.array([[l_points_prime[0][k]],
    #                             [l_points_prime[1][k]]])
    #
    #     l_prime = computeLine(F, point, prime=True)
    #
    #     plotInterestPoint(point_prime)
    #     plotEpipolarLines(l_prime, e_prime, point_prime)
    #
    # #plt.legend()
    #
    # plt.show()
    #
    ############################################################################################
    # # validation de l'algorithme sur dataset maison: mesure de distance
    # l_points_test=np.array([[1184.58, 1415.35, 1634.46, 2505.25, 446.32, 892.20, 1229.42, 1318.12],
    #                         [2576.34, 2376.12, 2182.66, 1428.94, 1399.13, 1929.99, 2336.44, 2444.86]])
    #
    # l_points_test_prime=np.array([[989.58, 1138.59, 1296.89, 2102.25, 521.53, 781.57, 1010.93, 1076.77],
    #                               [2253.69, 2131.19, 2003.83, 1339.74, 1399.13, 1759.65, 2080.29, 2173.74]])
    #
    # l_normalized_test, T=normalisation(l_points_test)
    # l_normalized_test_prime, _=normalisation(l_points_test_prime)
    # A_test=computeAbis(l_normalized_test, l_normalized_test_prime)
    # F_test=fundamentalMatrixSvd(A_test,T)
    # e_test=computeFirstEpipole(F_test)
    # e_test_prime=computeSecondEpipole(F_test)
    # P_test = np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix of the first camera, identity
    # P_test_prime = computePfromF(F_test, e_test, e_test_prime)
    #
    # # couple (2cm et 12 cm) sur ma règle
    # point1_test2cm, point2_test2cm=np.array([[1301.32], [2475.32]]), np.array([[1061.12], [2193.94]])
    # point1_test12cm, point2_test12cm=np.array([[1846.98], [1997.79]]), np.array([[1466.79], [1865.82]])
    #
    # # couple (0cm et 16,5cm) sur règle rayane
    # point1_testR0cm, point2_testR0cm = np.array([[446.30], [1399.22]]), np.array([[521.53], [1399.25]])
    # point1_testR16_5cm, point2_testR16_5cm = np.array([[1296.59], [2417.48]]), np.array([[1057.88], [2147.63]])
    #
    # X_test2cm=point_triangulation(point1_test2cm,point2_test2cm, P_test, P_test_prime)
    # X_test12cm=point_triangulation(point1_test12cm, point2_test12cm, P_test, P_test_prime)
    #
    # X_testR0cm=point_triangulation(point1_testR0cm, point2_testR0cm, P_test, P_test_prime)
    # X_test16_5cm=point_triangulation(point1_testR16_5cm, point2_testR16_5cm, P_test, P_test_prime)
    #
    # # X_test2cm=X_test2cm / X_test2cm[0][-1]
    # # X_test12cm=X_test12cm / X_test12cm[0][-1]
    # #
    # # X_testR0cm=X_testR0cm / X_testR0cm[0][-1]
    # # X_test16_5cm=X_test16_5cm / X_test16_5cm[0][-1]
    #
    #
    # print('This is X2cm', X_test2cm)
    # print('This is X12cm', X_test12cm)
    #
    # print('This is XR0cm', X_testR0cm)
    # print('This is XR16_5cm', X_test16_5cm)
    #
    # print("vector's norm couple 2 and 12: ",np.sqrt((X_test2cm[0][0]-X_test12cm[0][0])**2+(X_test2cm[0][1]-X_test12cm[0][1])**2+(X_test2cm[0][2]-X_test12cm[0][2])**2))
    # print("vector's norm couple R 0 and 16,5: ", np.sqrt((X_testR0cm[0][0]-X_test16_5cm[0][0])**2+(X_testR0cm[0][1]-X_test16_5cm[0][1])**2+(X_testR0cm[0][2]-X_test16_5cm[0][2])**2))
    #
    # plotImages('IMG_2474.jpg')
    # plotEpipoles(e_test)
    # plotInterestPoint(point1_test2cm)
    # plotInterestPoint(point1_test12cm)
    #
    # plotImages('IMG_2475.jpg')
    # plotEpipoles(e_test_prime)
    # plotInterestPoint(point2_test2cm)
    # plotInterestPoint(point2_test12cm)
    #
    # plt.show()

    ############################################################################################
    # 2ème batterie de test: localisation
    # l_points_test = np.array([[1262.98, 1244.85, 314.44, 1358.04, 1066.83, 1473, 954.08, 370.28],
    #                           [1877.19, 1941.58, 1639.13, 1716.17, 1819.08, 1818.83, 1709.92, 1706.65]])
    #
    # l_points_test_prime = np.array([[1485.47, 1340.44, 2216.94, 2522, 1605.33, 1644, 2348.31, 2174.81],
    #                                 [2382.28, 2378.72, 1917.25, 2655.50, 2249.33, 2529.67, 2326.81, 1988.69]])
    #
    # l_normalized_test, T = normalisation(l_points_test)
    # l_normalized_test_prime, _ = normalisation(l_points_test_prime)
    # A_test = computeAbis(l_normalized_test, l_normalized_test_prime)
    # F_test = fundamentalMatrixSvd(A_test, T)
    # e_test = computeFirstEpipole(F_test)
    # e_test_prime = computeSecondEpipole(F_test)
    # P_test = np.hstack((np.eye(3), np.zeros((3, 1))))  # camera matrix of the first camera, identity
    # P_test_prime = computePfromF(F_test, e_test, e_test_prime)
    #
    # # couple (2cm et 12 cm) sur ma règle
    # point1_testLed, point2_testLed = np.array([[1244.85], [1941.58]]), np.array([[1340.44], [2378.72]])
    #
    # X_testLed = point_triangulation(point1_testLed, point2_testLed, P_test, P_test_prime)
    # X_testLed=X_testLed/X_testLed[0][-1]
    #
    # # X_test2cm=X_test2cm / X_test2cm[0][-1]
    # # X_test12cm=X_test12cm / X_test12cm[0][-1]
    # #
    # # X_testR0cm=X_testR0cm / X_testR0cm[0][-1]
    # # X_test16_5cm=X_test16_5cm / X_test16_5cm[0][-1]
    #
    # print('This is X', X_testLed)
    #
    # plotImages('IMG_2476.jpg')
    # plotEpipoles(e_test)
    # plotInterestPoint(point1_testLed)
    #
    # plotImages('IMG_2477.jpg')
    # plotEpipoles(e_test_prime)
    # plotInterestPoint(point2_testLed)
    #
    # plt.show()


    # ############################################################################
    # comparasion au code de reference

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

    