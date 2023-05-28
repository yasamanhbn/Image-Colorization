import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import color
np.random.seed(1)
HEIGHT=224
WIDTH=224
import torch

def preprocessing(ImagePath):
    print("start reading images")
    X_img = []
    for imageDir in os.listdir(ImagePath):
        img = cv2.cvtColor(cv2.imread(ImagePath + '/' + imageDir), cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32)
        resized_img_lab = cv2.resize(img, (WIDTH, HEIGHT)) # resize image to network input size
        img_lab = torch.FloatTensor(np.transpose(color.rgb2lab(np.array(resized_img_lab)), (2, 0, 1)))
        X_img.append(img_lab)
    print("preprocessing has just finished")
    return X_img