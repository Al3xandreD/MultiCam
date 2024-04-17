import numpy as np
import matplotlib.pyplot as plt
import cv2

def cleaning(img,k):
    '''
    Clean noise on an image using a gaussian filter
    :param img: gray level image
    :param k: size of kernel for convolution
    :return:
    '''

    img_filt=cv2.GaussianBlur(img,(k,k), 0)
    return img_filt

def stereoCorrespondance(img_left,img_right, i_pixel_left,f_left,f_right):
    '''
    Determines the pixel in the right frame corresponding, to a given pixel in the left frame
    :param img_left:
    :param img_right:
    :param i_pixel_left: indexes of the left pixel
    :param f_left: gray level of the left image
    :param f_right: gray level of the right image
    :return: pixel_right
    '''

    i, j= i_pixel_left[0], i_pixel_left[1]
    liste_min=[]    # contient les compansantes des disparités minimales

    # TODO initialiser dc_min et dc_max
    disp=np.array([[i-i], [0-j]])
    for v in range(1,img_right.shape[1]): # pour toutes les colonnes
        disp=np.array([[i-i], [v-j]])

        if np.min(disp)<dc_min:
            dc_min = disp[0][1]
        if np.max(disp)>dc_max:
            dc_max=disp[1][1]

    dc_min=np.min()
    Zd=np.zeros((1,dc_max-dc_min))


def knnCorrespondance(img_left,img_right):
    '''

    :param img_left:
    :param img_right:
    :return:
    '''

    # Initialiser l'extracteur de points d'intérêt
    sift = cv2.SIFT_create()

    # Trouver les points d'intérêt et descripteurs pour chaque image
    keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

    # Initialiser le matcher de correspondance
    bf = cv2.BFMatcher()

    # Faire correspondre les descripteurs entre les images gauche et droite
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    # Appliquer le ratio test pour filtrer les correspondances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # Dessiner les correspondances sur une nouvelle image
    img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None)

    # Afficher l'image avec les correspondances
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def flannMatching(img_left,img_right):
    sift = cv2.SIFT_create()

    keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

    # Paramètres pour le matcher FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Nombre de vérifications pour l'algorithme de recherche

    # Créer un objet FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Faire correspondre les descripteurs des deux images
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

    # Filtrer les correspondances en utilisant le ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Afficher les correspondances
    img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histo(img):
    '''
    Compute the histogram of a given image
    :param img: grey level image
    :return:
    '''

    hist=cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

def compHisto(hist_left, hist_right):
    '''
    Compares the nature of two histograms
    :param hist_left:
    :param hist_right:
    :return: score
    '''

    return np.sum(np.abs(hist_left-hist_right), axis=0)

def newStereoCorrespondance(img_left,img_right, i_pixel_left, f_left, f_right):
    '''

    :param img_left:
    :param img_right:
    :param i_pixel_left:
    :param f_left:
    :param f_right:
    :return:
    '''

    # TODO:definir le mask
    # TODO deplacer le mask
    # TODO comparer les histogrammes

    mask_size=3
    for i in range(img_right.shape[0]): # ligne
        for j in range(img_right.shape[1]): # colonne
            mask=img_right[i:i+mask_size-1,j:j+mask_size-1]
            hist_right=histo(mask)
            hist_left=histo(img_left[i_pixel_left[0]-1:i_pixel_left[0]+1,i_pixel_left[1]-1:i_pixel_left[1]+1])
            compHisto(hist_left,hist_right)

            # if errhisto<max_errhisto:
            #     i_pixel_right=[i,j]


if __name__=='__main__':

    # donner pour image_left le crop de la box de la personne pour la camera maitre
    # donner pour image_droite la frame de la camera esclave

    img_left=cv2.imread('../../RnD/Computer Vision/Multiview/checkboard_left.jpg', 0)
    img_right=cv2.imread('../../RnD/Computer Vision/Multiview/checkboard_right.jpg', 0)

    hist_left=histo(img_left)
    hist_right=histo(img_right)
    erreur=compHisto(hist_left, hist_right)
    print(erreur)

    knnCorrespondance(cleaning(img_left,3), cleaning(img_right,3))
    #flannMatching(img_left,img_right)

    # TODO: se baser sur vitesse, orietation, histogramme de niveau de gris