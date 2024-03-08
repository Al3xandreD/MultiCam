import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
The purpose of this script is to make easier the process of creation 
of heterogeneous datasets for binary classification by concatenating 
image from class 1 to image to class 2 and save the results to a specified
dataset folder. 

The images are concatenated in a random way to avoid non-generalisation problems.
That is the purposes of random side-keys that can be stored in a separated folder
"""

DATASET1_FOLDER_PATH = r"..\Fireman\0_dataset1"
DATASET2_FOLDER_PATH = r"..\Fireman\0_dataset2"

OUT_FOLDER_PATH = r"..\Fireman\1_mixed_train_dataset"
side_key_path = OUT_FOLDER_PATH + "/side_keys"

# Supported images extension :
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

dataset1 = []
dataset2 = []
random_keys = []


def load_images_to_arrays(folder_path):
    image_arrays = []

    # Check folder path
    if not os.path.exists(folder_path):
        print(f"Le dossier '{folder_path}' n'existe pas.")
        return

    # Loop over folder files
    for filename in os.listdir(folder_path):
        # Check file extension
        if any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS):

            # Build the entire image file path
            file_path = os.path.join(folder_path, filename)

            # Read image with OpenCV
            image = cv2.imread(file_path)

            if image is not None:
                image_arrays.append(image)
            else:
                print(f"Échec du chargement de l'image : {file_path}")

    return image_arrays


dataset1 = load_images_to_arrays(DATASET1_FOLDER_PATH)
dataset2 = load_images_to_arrays(DATASET2_FOLDER_PATH)
n = min(len(dataset1), len(dataset2))

# Run concatenation
for i in range(n):

    # Images
    im1 = dataset1[i]
    im2 = dataset2[i]

    # Get images dimensions
    size1 = im1.shape[:2]
    size2 = im2.shape[:2]
    common_size = (640, 480)

    # Resize
    image1_resized = cv2.resize(im1, common_size)
    image2_resized = cv2.resize(im2, common_size)

    # Concatenate / combine
    random_key = np.random.randint(0, 4)
    random_axis = int(random_key > 1)

    if random_key % 2:
        tmp = image1_resized
        image1_resized = image2_resized
        image2_resized = tmp

    concatenated_image = np.concatenate((image1_resized, image2_resized), axis=random_axis)

    # Save concatenated images
    im_name = f"{i:d}_{random_key:d}" if (side_key_path is None or side_key_path == "") else str(i)
    cv2.imwrite(f"{OUT_FOLDER_PATH:s}/{im_name:s}.png", concatenated_image)

    # Save side-keys in separate folder
    if not (side_key_path is None or side_key_path == ""):
        if not os.path.exists(side_key_path):
            os.makedirs(side_key_path)
        with open(f"{side_key_path:s}/{im_name:s}_side_key.txt", 'w') as f:
            f.write(f"{random_key:d}")
