import os
import subprocess
import tensorflow as tf

def move2directory(pathYolo):
    '''
    Navigating to a defined path
    :param pathYolo:
    :return:
    '''
    file_directory = os.path.dirname(pathYolo)
    os.chdir(file_directory)  # navigating to yolo position


def detection(pathDatabase, pathWeights):
    '''
    Applies Yolo inference to an image
    :param pathImage: path of image
    :param pathYolo: path of yolo detect files
    :return:
    '''

    command = "python yolov7/detect.py --source "+"'"+pathDatabase+"'"+" --weights "+"'"+pathWeights+"'"+" --save-txt"
    resultYolo = subprocess.run(command, shell=True)

    print(resultYolo.stdout)

def detectionUltralytics(pathDatabase):
    '''
    Applies Yolo inference to a database usin Yolo ultralytics
    :param pathDatabase:
    :param pathWeights:
    :return:
    '''

    command="python engine/predictor.py model=yolov8n.pt source="+pathDatabase
    resultYolo = subprocess.run(command, shell=True)
    print(resultYolo.stdout)

def training(pathTrain, pathData_train, pathWeights, epochs):
    '''
    Trains yolo according to the given dataset
    :param pathTrain: path to train.py
    :param pathData_train: path to dataset
    :param pathWeights: paths to reference weights
    :return:
    '''
    # TODO ajouter la version mps pour supporter apple silicon
    command="python "+"'"+pathTrain+"'" +" --workers 8 --device mps --epochs " + epochs + " --data " + "'"+pathData_train+"'"+" --img 640 --weights "+"'"+pathWeights+"'"+""
    command2="python "+"'"+pathTrain+"'"+" --workers 8 --img 640 --epochs "+ epochs +" --data "+"'"+pathData_train+"'"+" --weights "+ "'"+pathWeights+"'"
    resultTrain = subprocess.run(command, shell=True)

    print(resultTrain.stdout)

def execResult(pathWeight):
    '''
    Executes yolo according to a model
    :param pathWeight: path to weights
    :return:
    '''

    command="python yolov7/detect.py --source 0 --weights "+"'"+pathWeight+"'"
    resultExec = subprocess.run(command, shell=True)

    print(resultExec.stdout)

if __name__=='__main__':

    label_data=True
    re_training=False
    exec_result=False

    # path for detection and draw boxes
    pathYolo = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/Saves/yolov7"
    pathWeights = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/Saves/yolov7/yolov7.pt"    # standard weights
    pathDatabase = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/Computer Vision RnD/Trains/archive"

    # path for training
    pathData_train = "MultiCamTraining/data.yaml"
    pathTrain = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/Saves/yolov7/train.py"

    # path for applying retrained model
    pathBest="resultWeights/best.pt"

    # pathUltralytics
    pathYoloUltralytics="/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/Saves/ultralytics-main/ultralytics/engine"

    if label_data:
        #move2directory(pathYoloUltralytics)
        #detectionUltralytics(pathDatabase)
        move2directory(pathYolo)
        detection(pathDatabase,pathWeights)

    if re_training:
        move2directory(pathYolo)
        training(pathTrain, pathData_train, pathWeights)

    if exec_result:
        move2directory(pathYolo)
        execResult(pathBest)





