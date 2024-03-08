import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import output_to_keypoint
from profiling.profiling_recorder import ProfilingRecorder

""" Multi Pose Yolo based estimator """

class PoseEstimator:

    def __init__(self, device_name, poseweights="poseestimation/weights/yolov7-w6-pose.pt"):
        self.device = select_device(device_name)  # select device
        strip_optimizer(self.device, poseweights)
        half = self.device.type != 'cpu'
        self.model = attempt_load(poseweights, map_location=self.device)  # Load model
        _ = self.model.eval()
        self.model_stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.profiler = ProfilingRecorder()

    def detect(self, input_frame):
        # This model works with an unsigned 8-bit integer numpy array,
        # so we need to convert the float64 coded image onto 8bit unsigned :
        input_frame = np.uint8(input_frame)

        # This model works with images having size multiple of model_stride = gcd = 64,
        # so we need to resize the image to the closest multiple of 64  :
        adapted_frame = input_frame
        # adapted_frame = gcd_resize(input_frame, gcd)
        frame_out = adapted_frame

        orig_image = adapted_frame  # store frame
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
        # image = orig_image
        image = letterbox(image, (adapted_frame.shape[1]), stride=self.model_stride, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        image = image.to(self.device)  # convert image data to device
        image = image.float()  # convert image to float precision (cpu)

        self.profiler.start()
        with torch.no_grad():  # get predictions
            output_data, _ = self.model(image)

        output_data = non_max_suppression_kpt(output_data,  # Apply non-max suppression
                                              0.25,  # Conf. Threshold.
                                              0.65,  # IoU Threshold.
                                              nc=self.model.yaml['nc'],  # Number of classes.
                                              nkpt=self.model.yaml['nkpt'],  # Number of keypoints.
                                              kpt_label=True)
        output_frame = image
        raw_results = output_data
        keypoints = output_to_keypoint(output_data)
        self.profiler.measure()


        return output_frame, raw_results, keypoints, self.profiler
