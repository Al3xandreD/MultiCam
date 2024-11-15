from math import sqrt
from time import time

import cv2

import hpc.consts as c
from hpc.core.skeleton import Skeleton


# class to proceed frames, remembering previous skeletons ec.
class Frame:
    def __init__(self, model=None, dynModel=None, live=True):
        self.skeletons = []
        self.model = model
        self.dynModel = dynModel
        self.lastSkeletonId = 0
        self.live = live
        self.prevTime = time()
        self.frameTime = 0

    # function returns probabilities list of each pose for each given human (with skeleton id)
    # humans is list of humans with list of keypoints for every human and probability of this keypoint (0, when it's lower than threshold)
    def proceedFrame(self, humans):
        if self.live:
            self.frameTime = min(self.prevTime - time(), c.maxFrameTime)
            self.prevTime = time()
        if not humans:
            self.skeletons = []
            return []
        newSkeletons = []  # here will be all detected skeletons
        for human in humans:
            self.__proceedHuman(human, newSkeletons)
        # self.skeletons = [ s for s in newSkeletons if s is not None ]   # we save only existing skeletons
        self.skeletons = newSkeletons
        poses = []
        if self.dynModel is None:  # normal mode
            for skeleton in newSkeletons:
                poses.append([classifyPose(skeleton, self.model), skeleton.getSkeletonId()])
        else:
            for skeleton in newSkeletons:  # hybrid mode
                if skeleton.getPointsDistance(c.distancePoints) < c.statDynThreshold:
                    poses.append([classifyPose(skeleton, self.model), skeleton.getSkeletonId()])
                else:
                    poses.append([classifyPose(skeleton, self.dynModel), skeleton.getSkeletonId()])
        return poses

    def __proceedHuman(self, human, newSkeletons):
        sameSkeletonProb = []  # probability, that human is 'i' skeleton
        bb = getBoundingBox(human)
        minDelta = getMinDelta(bb)
        for skeleton in self.skeletons:
            sameSkeletonProb.append(skeleton.compareSkeleton(human, minDelta))
        if len(sameSkeletonProb) != 0:
            maxProb = max(sameSkeletonProb)
        else:
            maxProb = 0
        if maxProb >= c.probThreshold:  # skeletons are the same human
            i = sameSkeletonProb.index(maxProb)
            self.skeletons[i].updateSkeleton(human, bb)  # update skeleton
            newSkeletons.append(self.skeletons[i])  # add skeleton to new skeletons
            self.skeletons.pop(i)  # skeleton cannot be compared again
        else:
            newSkeletons.append(Skeleton(human, self.lastSkeletonId, bb))  # make new skeleton if there is no similar skeleton
            self.lastSkeletonId = self.lastSkeletonId + 1

    # function takes detected humans keypoints and return skeleton image for each human
    # this is equivalent to proceedFrame, but for creating dataset
    def getSkeletons(self, humans):
        if not humans:
            self.skeletons = []
            return []
        newSkeletons = []
        for human in humans:
            self.__proceedHuman(human, newSkeletons)
        # self.skeletons = [ s for s in newSkeletons if s is not None ]
        self.skeletons = newSkeletons
        images = []
        for skeleton in self.skeletons:
            # if skeleton is not None:
            images.append([skeleton.getSkeletonImg(), skeleton.getSkeletonId()])
            # else:
            #     images.append( None )
        return images

    # function does the same as getSkeletons() but it returns last skeleton keypoints instead of image
    def getKeypoints(self, humans):
        if not humans:
            return []
        newSkeletons = []
        for human in humans:
            self.__proceedHuman(human, newSkeletons)
        self.skeletons = newSkeletons
        keypoints = []
        for skeleton in self.skeletons:
            keypoints.append([skeleton.getSkeletonKeypoints(), skeleton.getSkeletonId()])
        return keypoints


# functions classify pose and returns probabilities of poses
def classifyPose(skeleton, model):
    return model.predict(
        (cv2.rotate(skeleton.getSkeletonImg(), cv2.ROTATE_90_CLOCKWISE)).reshape(-1, c.keypointsNumber,
                                                                                 c.framesNumber, 3))


# returns list [ [ maxW, minW ], [ maxH, minH ], [ maxD, minD ] ]
def getBoundingBox(keypoints):
    maxmins = [[0, c.frameWidth], [0, c.frameHeight], [0, 65535]]
    for keypoint in keypoints:
        if keypoint[3] != 0.0:  # if keypoint detected
            for i in range(3):
                if keypoint[i] > maxmins[i][0]:
                    maxmins[i][0] = keypoint[i]
                if keypoint[i] < maxmins[i][1]:
                    maxmins[i][1] = keypoint[i]
    return maxmins


def getMinDelta(boundingBox):
    return c.maxDeltaCoefficient * sqrt(pow(boundingBox[0][0] - boundingBox[0][1], 2) +
                                        pow(boundingBox[1][0] - boundingBox[1][1], 2))
