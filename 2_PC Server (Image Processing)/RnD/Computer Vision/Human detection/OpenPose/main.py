# REF : https://github.com/quanhua92/human-pose-estimation-opencv

import cv2
import urllib.request
import numpy as np
import time
import argparse
import socket
from urllib.error import URLError, HTTPError

import requests

def url_ok(url, timeout = 1):
    r = requests.head(url, timeout=timeout)
    return r.status_code == 200

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height



global img
enableDetection = 0



######################## Main functions definitions ########################
def RunDetection(frame):
    global frameHeight, frameWidth
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    global out
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    return (len(BODY_PARTS) == out.shape[1])
        

def PostProcess():
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

def check_url_availability(url, timeout=2):
    try:
        # Configurez une requête avec un timeout spécifié
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request, timeout=timeout)
        
        # Si la requête réussit, l'URL est disponible
        print(f"L'URL {url} est disponible.")
        return True
    except HTTPError as e:
        # Si le serveur retourne un code d'erreur HTTP
        print(f"L'URL {url} a retourné un code d'erreur HTTP {e.code}.")
        return False
    except URLError as e:
        # Si une exception d'URL se produit (par exemple, connexion refusée)
        print(f"Une exception URLError s'est produite : {e.reason}.")
        return False
    except Exception as e:
        # Gérer d'autres exceptions qui pourraient se produire
        print(f"Une exception s'est produite : {e}.")
        return False

########################################################################################################################









#################################################### Program ####################################################

    
#########  Init and Open camera video stream #########
url = "http://172.19.147.182/cam-lo.jpg" #cam-lo  #cam-mid #cam-hi
cv2.namedWindow("Live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Create a VideoCapture object
# cap = cv2.VideoCapture(url)                 

# Check if the IP camera stream is opened successfully
# if not cap.isOpened():
#     print("Failed to open the IP camera stream")
#     exit()
###########################################################




########## Detection Init #############     
if(enableDetection):
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
#######################################



#Init frame
frame = np.array(np.eye(80), dtype=np.uint8)

########## Video and Detection #########
while True:


    

    ### Read a frame from the video stream ###
    #Meth:1
    # try:
    #     if (url_ok(url)):
    #         img_resp = urllib.request.urlopen(url)
    #         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    #         img = cv2.imdecode(imgnp, -1)
    #         img = cv2.transpose(img)
    #         frame = img
    #         hasFrame = True
    # except Exception as e:        
    #     print(e)

    #Meth:2
    try:
        if (url_ok(url, 1)):
            cap = cv2.VideoCapture(url)                 
            hasFrame, frame = cap.read()
    except Exception as e:        
        print(e)

    ### Try detection and process frame with results ###
    if(enableDetection and hasFrame):
        if (RunDetection(frame)):
            PostProcess()
        

    ### Display current frame with its detection results ### 
    cv2.imshow('live Cam Testing', frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

