
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# loading images
# im1=np.array(Image.open('imaged/001.jpg'))
# im2=np.array(Image.open('imaged/001.jpg'))
#
# # loading 2D points for each images
# points2D=[np.loadtxt('2D/00'+str(i+1)+'.corners').T for i in range (3)]
#
# # loading 3D points
# point3D=np.loadtxt('3D/p3d').T
#
# # loading correspondences
# corr=np.genfromtxt('2D/nviews-corners', dtype='int', missing_values='*')
#
# # loading cameras
# P=[camera.Camera(np.loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]
#
# X_s=np.vstack(point3D,np.ones(point3D.shape[1]))
# x=P[0].np.project(X_s)
#
# # plotting on view1
# plt.figure()
# plt.imshow(im1)
# plt.plot(points2D[0][0],points2D[0][1],'*')
# plt.axis('off')
#
# plt.figure()
# plt.imshow(im1)
# plt.plot(x[0], x[1], 'r.')
# plt.axis=('off')
#
# plt.show()
#
# # plotting in 3D
# fig=plt.figure
# ax=fig.gca(projection="3d")
#
# X,Y,Z=axes3d.get_test_data(0.25)
# ax.plot(X.flatten(), Y.flatten(), Z.flatten(), 'o')
# ax.show()
def computeA(l_points):
    '''
    Compute the A matrix, necessary to compute the fundamental matrix
    :param l_points: array of points
    :return: A
    '''
    A = np.zeros((l_points.shape[1], 9))
    for k in range(l_points.shape[1]):
        A[k] = np.array([l_points[0][k][0][0] * l_points[1][k][0][0], l_points[0][k][0][0] * l_points[1][k][1][0],
                         l_points[0][k][0][0], l_points[0][k][1][0] * l_points[1][k][0][0],
                         l_points[0][k][1][0] * l_points[1][k][1][0], l_points[0][k][1][0], l_points[1][k][0][0],
                         l_points[1][k][1][0], 1])
    return A

def fundamentalMatrixLsq(l_points):
    """
    Computes the fundamental matrix using least-square method
    :param l_points: 2 line array, first line is for the first camera and second line is for the second camera
    :return: Fundamental matrix
    """
    A=computeA(l_points)
    y=np.zeros((l_points.shape[0],9))
    return np.linalg.inv(A.T@A)@A.T@y

def fundamentalMatrixSvd(l_points):
    '''
    Computes the fundamental matrix using singular value decomposition method
    :param l_points: 2 line array, first line is for the first camera and second line is for the second camera
    :return: Fundamental matrix
    '''

    A=computeA(l_points)
    U,S,V=np.linalg.svd(A)
    F=V[-1].reshape(3,3)

    # applying constraint on F rank 2
    # by zeroing out the last singular value
    U,S,V=np.linalg.svd(F)
    S[2]=0
    F=U@(np.diag(S)@V)
    return F

def computeFirstEpipole(F):
    '''
    Computes the epipole of the first camera according to least-square method, Fe=0
    :param F: Fundamental matrix
    :return: e, epipole
    '''

    y=np.zeros((3,1))
    return np.linalg.inv(F.T@F)@F.T@y

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

    # TODO: définir la fonction permettant de calculer P si la camera est calibrée
    return None
def computePfromF(F):
    '''
    Computes the second camera matrix if uncalibrated
    :param F:
    :return:
    '''
    e=computeFirstEpipole(F) # epipole, column vector
    e_prime=computeSecondEpipole(F)
    skew_e=np.array([[0, -e[2][0], e[1][0]],
                     [e[2][0], 0, -e[0][0]],
                     [-e[1][0], e[0][0], 0]])   # skew matrix associated with the epipole
    Pprime=np.hstack((skew_e@F,e_prime))
    return Pprime


def point_triangulation(point_x, point_xprime, P, Pprime):
    '''
    Triangulates a point from two image points via least-square method
    :param point_x: first point on first camera
    :param point_xprime: second point on second camera
    :param P: first camera matrix
    :param Pprime: second camera matrix
    :return: X coordinates of the triangulated point
    '''

    A1=np.array([[point_x[1][0]@P[2]-P[1]],
                 P[0]-point_x[0][0]@P[2]])
    A2=np.array([[point_xprime[1][0]@Pprime[2]-Pprime[1]],
                 Pprime[0]-point_xprime[0][0]@Pprime[2]])
    A=np.vstack((A1,A2))
    y=np.zeros((4,1))
    return np.linalg.inv(A.T@A)@A.T@y

if __name__=='__main__':
    x_1 = np.array([2, 3]).reshape(2,1)
    x_1_prime = np.array([5, 8]).reshape(2, 1)    # point fictifs

    x_2 = np.array([5, 9]).reshape(2, 1)
    x_2_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_3 = np.array([2, 31]).reshape(2, 1)
    x_3_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_4 = np.array([8, 10]).reshape(2, 1)
    x_4_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_5 = np.array([12, 7]).reshape(2, 1)
    x_5_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_6 = np.array([5, 9]).reshape(2, 1)
    x_6_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_7 = np.array([1, 7]).reshape(2, 1)
    x_7_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    x_8 = np.array([23, 12]).reshape(2, 1)
    x_8_prime = np.array([5, 8]).reshape(2, 1)  # point fictifs

    l_points = np.array([[x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8],
                         [x_1_prime, x_2_prime, x_3_prime, x_4_prime, x_5_prime, x_6_prime, x_7_prime,x_8_prime]])


