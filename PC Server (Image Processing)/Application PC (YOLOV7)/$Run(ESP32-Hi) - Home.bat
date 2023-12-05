::Faire tourner avec la webcam
::python detect.py --source 0 --weights "C:\Users\rayou\Documents\ENSTA\Projet MultiCam\repo\yolov7\yolov7.pt"

::Faire tourner avec webcam distante(Ex : ESP32)
::python detect.py --source "http://192.168.0.23/cam-hi.jpg" --weights "C:\Users\rayou\Documents\ENSTA\Projet MultiCam\repo\yolov7\yolov7.pt"
python detect.py --source "C:\Users\MC\Desktop\github_Multicam\MultiCam\PC Server (Image Processing)\Application PC (YOLOV7)\streams.txt" --weights "C:\Users\MC\Desktop\github_Multicam\MultiCam\PC Server (Image Processing)\Application PC (YOLOV7)\yolov7.pt"
