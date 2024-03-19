import os
import subprocess

def move2directory(pathYolo):
    '''
    Navigating to a defined path
    :param pathYolo:
    :return:
    '''
    file_directory = os.path.dirname(pathYolo)
    os.chdir(file_directory)  # navigating to yolo position
def detection(pathImage, pathWeights):
    '''
    Applies Yolo inference to an image
    :param pathImage: path of image
    :param pathYolo: path of yolo detect files
    :return:
    '''

    command = "python detect.py --source "+"'"+pathImage+"'"+" --weights "+"'"+pathWeights+"'"+" --save-txt"
    resultYolo = subprocess.run(command, shell=True)

    print(resultYolo.stdout)

def training(pathTrain, pathData_train, pathWeights):
    '''
    Training yolo according to the given dataset
    :param pathTrain: path to train.py
    :param pathData_train: path to dataset
    :param pathWeights: paths to reference weights
    :return:
    '''
    # command="python "+"'"+pathTrain+"'" +" --workers 8 --device mps --batch-size 32 --data " + "'"+pathData_train+"'"+" --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml"
    command2="python "+"'"+pathTrain+"'"+" --workers 8 --img 640 --epochs 3 --data "+"'"+pathData_train+"'"+" --weights "+ "'"+pathWeights+"'"
    resultTrain = subprocess.run(command2, shell=True)

    print(resultTrain.stdout)

def execResult(pathWeight):
    '''
    Executes yolo according to a model
    :param pathWeight: path to weights
    :return:
    '''

    command="python yolov7/detect.py --source 0 --weights "+"'"+pathWeight+"'"
    "detect.py --source 0 --weights '/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/Saves/yolov7/resultWeights/best.pt' --device mps"

    resultExec = subprocess.run(command, shell=True)

    print(resultExec.stdout)

if __name__=='__main__':

    label_data=False
    re_training=True
    exec_result=False

    # path for detection and draw boxes
    pathYolo = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/RnD/Computer Vision/Training/Auto_dataset_tools"
    pathWeights = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/RnD/Computer Vision/Training/Auto_dataset_tools/yolov7.pt"
    pathDatabase = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/RnD/Computer Vision/Training/Fireman/dataset"

    # path for training
    pathData_train = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/RnD/Computer Vision/Training/Fireman/MultiCamTraining/data.yaml"
    pathTrain = "/Users/alexandredermouche/Documents/Alexandre /Cours/ENSTA/2A/STIC/MultiCam/2_PC Server (Image Processing)/RnD/Computer Vision/Training/Auto_dataset_tools/train.py"

    # path for executing
    pathBest="resultWeights/best.pt"

    if label_data:
        #move2directory(pathYolo)
        detection(pathDatabase, pathWeights)

    if re_training:
        move2directory(pathYolo)
        training(pathTrain, pathData_train, pathWeights)

    if exec_result:
        move2directory(pathYolo)
        execResult(pathBest)





