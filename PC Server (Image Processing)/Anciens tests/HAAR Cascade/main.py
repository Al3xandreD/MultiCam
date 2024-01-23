#Ref : https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

import cv2
import urllib.request
import numpy as np
import time
 
global img
enableDetection = False

def RunDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global detectionResults
    detectionResults = fullbody_cascade.detectMultiScale(gray, 1.3, 5)
    return len(detectionResults)
        

def PostProcess():
    for (x,y,w,h) in detectionResults:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



#########  Init and Open camera video stream #########
url = "http://192.168.43.144/cam-mid.jpg" #cam-lo  #cam-mid #cam-hi
cv2.namedWindow("Live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()
###########################################################



########## Detection Init #############
fullbody_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#######################################



########## Video and Detection #########
while True:

    # Read a frame from the video stream
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    # ret, frame = cap.read()
    img = cv2.imdecode(imgnp, -1)
    img = cv2.transpose(img,img)
    
    ## Try detection and process frame with results
    if(enableDetection):
        if (RunDetection(img)):
            PostProcess()
        

    ### Display current frame with its detection results ### 
    cv2.imshow('live Cam Testing', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

