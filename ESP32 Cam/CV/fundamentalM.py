import numpy as np
def compute_fundamental(x1, x2):
    """
    Computes the fundamental matrix according to the eight point algorithm
    :param x1: forms a correspondence pair with x2
    :param x2:
    :return: F
    """

    n=x1[1].shape
    if x2[1].shape!=n:
        raise ValueError("Number of poits don't match")

    # Matrix A
    A=np.zeros((n,9))
    for i in range(n):     # for all lines
        A[i]=[x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
              x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
              x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]
    U,S,V=np.linalg.svd(A)
    F=V[-1].reshape(3,3)



