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

    # img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches[:10], None)
    # point1=keypoints_left[good_matches[0].queryIdx].pt
    # print(point1)
    # point2 = keypoints_right[good_matches[0].trainIdx].pt
    # cv2.circle(img_matches, (int(point1[0]), int(point1[1])), 20, (255, 0, 0), 10, cv2.FILLED)
    # cv2.circle(img_matches, (int(point2[0] + img_left.shape[0]), int(point2[1])), 20, (255, 0, 0), 10, cv2.FILLED)
    # cv2.imshow('Matches', img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return good_matches, keypoints_left, keypoints_right

def match2pixel(good_matches, keypoints_left, keypoints_right):
    '''
    Converts a list of matches to the corresponding pixels
    :param good_matches:
    :param keypoints_left:
    :param keypoints_right:
    :return:
    '''

    list_pixels=[]

    for match in good_matches:
        point_left = keypoints_left[match.queryIdx].pt  # coordinates pixels
        point_right = keypoints_right[match.trainIdx].pt

        list_pixels.append([point_left, point_right])

    return list_pixels

def computeWindow(human_pose, frame_shape):
    '''
    Defines the size of window research
    :param human_pose:
    :param frame_shape:
    :return:
    '''


    if human_pose is not None:
        # Separate keypoints positions and keypoints confidence
        kpts_xy = human_pose.reshape(-1, 3)[:, :-1] # keypoints position
        kpts_score = human_pose.reshape(-1, 3)[:, :-1]   # keypoint confidence

        bbox = [min(kpts_xy[:, 0]), min(kpts_xy[:, 1]), max(kpts_xy[:, 0]),
                     max(kpts_xy[:, 1])]    # bounding box

        delta_heigth=15 # for increasing window size for robustness
        delta_width=15

        if (bbox[3]-delta_heigth>0) and (bbox[1]+delta_heigth< frame_shape[1]) and(bbox[0]-delta_width>0) and (bbox[2]+delta_width<frame_shape[0]):
            up_left_point = (bbox[0]-delta_width, bbox[3]-delta_heigth)
            down_right_point = (bbox[2]+delta_width, bbox[1]+delta_heigth)

            return up_left_point, down_right_point

        else:
            up_left_point=(bbox[0],bbox[3])
            down_right_point=(bbox[2], bbox[1])

            return up_left_point, down_right_point

    else:
        return None

def findDescripteur(human_pose, good_matches, keypoints_left, keypoints_right, frame_shape):
    '''
    Finds the closest SIFT descriptors to the window of research of
    a given human_pose
    :param human_pose:
    :param good_matches:
    :param keypoints_left:
    :param keypoints_right:
    :return:
    '''

    up_left_point, down_right_point = computeWindow(human_pose, frame_shape)  # pose's window

    point_left=keypoints_left[good_matches[0].queryIdx].pt  # coordinates of first descriptor in left frame

    # initializing for loops
    norm_up_left=np.sqrt((up_left_point[0]-point_left[0])**2 + (up_left_point[1]-point_left[1])**2)  # distance between up left point and first descriptor
    norm_down_right=np.sqrt((down_right_point[0] - point_left[0]) ** 2 + (down_right_point[1] - point_left[1]) ** 2)

    # searching for closest descriptor to up left point
    for match in good_matches[1:]:
        point_left = keypoints_left[match.queryIdx].pt  # coordinates pixels
        if np.sqrt((up_left_point[0]-point_left[0])**2 + (up_left_point[1]-point_left[1])**2) < norm_up_left:
            norm_up_left=np.sqrt((up_left_point[0]-point_left[0])**2 + (up_left_point[1]-point_left[1])**2)
            nn_up_left_query=point_left # closest descriptor in the left frame for up left point
            nn_up_left_train=keypoints_right[match.trainIdx].pt # closest descriptor in the right frame for up left point

    # searching for closest descriptor to down right point
    for match in good_matches[1:]:
        point_left = keypoints_left[match.queryIdx].pt  # coordinates pixels
        if np.sqrt((down_right_point[0] - point_left[0]) ** 2 + (down_right_point[1] - point_left[1]) ** 2) < norm_down_right:
            norm_down_right = np.sqrt((down_right_point[0] - point_left[0]) ** 2 + (down_right_point[1] - point_left[1]) ** 2)
            nn_down_right_query = point_left  # closest descriptor in the left frame for down right point
            nn_down_right_train = keypoints_right[match.trainIdx].pt  # closest descriptor in the right frame for down right point

    return nn_up_left_train, nn_down_right_train

def findHuman(humanLeft, list_human_right, good_matches, keypoints_left, keypoints_right):
    '''
    Finds the corresponding human in the right frame for a
    given human in the left frame
    :param humanLeft:
    :return:
    '''

    nn_up_left_train,nn_down_right_train=findDescripteur(humanLeft, good_matches, keypoints_left, keypoints_right)
    best_norm=np.sqrt((list_human_right[0].center[0] - nn_up_left_train[0]) ** 2 + (list_human_right[0].center[1] - nn_up_left_train[1]) ** 2) + np.sqrt((list_human_right[0].center[0]-nn_down_right_train[0]) ** 2 + (list_human_right[0].center[1]-nn_down_right_train[0])**2)

    for human in list_human_right[1:]:
        # distance to up left point in right frame
        norm_up_left_train = np.sqrt((human.center[0] - nn_up_left_train[0]) ** 2 + (human.center[1] - nn_up_left_train[1]) ** 2)
        # distance to down right point in right frame
        norm_down_right_train = np.sqrt((human.center[0]-nn_down_right_train[0]) ** 2 + (human.center[1]-nn_down_right_train[0])**2)
        total_norm=norm_up_left_train + norm_down_right_train
        if total_norm<best_norm:
            best_norm=total_norm
            best_human_match=human  # human more likely to be in the research window associated to bbox points

    return best_human_match


if __name__=='__main__':

    # donner pour image_left le crop de la box de la personne pour la camera maitre
    # donner pour image_droite la frame de la camera esclave

    img_left=cv2.imread('checkboard_left.jpg', 0)
    img_right=cv2.imread('checkboard_right.jpg', 0)

    good_matches, keypoints_left, keypoints_right=knnCorrespondance(cleaning(img_left,3), cleaning(img_right,3))

    best_human_match=findHuman(humanLeft, list_human_right, good_matches, keypoints_left, keypoints_right)

