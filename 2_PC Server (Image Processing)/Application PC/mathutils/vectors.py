import cv2
import numpy as np
import torch
from torchvision import transforms



def euclidean_distance(v1, v2):
    return torch.sqrt(torch.sum((v1 - v2) ** 2))


def correlation(v1, v2):
    mean_v1 = torch.mean(v1)
    mean_v2 = torch.mean(v2)
    cov = torch.sum((v1 - mean_v1) * (v2 - mean_v2))
    std_v1 = torch.sqrt(torch.sum((v1 - mean_v1) ** 2))
    std_v2 = torch.sqrt(torch.sum((v2 - mean_v2) ** 2))
    return cov / (std_v1 * std_v2)