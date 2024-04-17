import

if __name__ == "__main__":
    # TODO launch GUI

    alert=False

    source1, source2=0,0
    myHandler=HandlerCamera(source1, source2)
    # handler contains calibrated cameras

    if alert==True:
        myHandler.start()
        while alert==True:
            left_frame, right_frame = myHandler.update()

            # TODO YOLO inference/Pose estimation
            # TODO Brute Force matching
            # TODO localisation

            # TODO A*

