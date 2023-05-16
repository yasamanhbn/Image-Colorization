import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import color
np.random.seed(1)
ImagePath = "F:/Master/Deep learning/HW/hw3/Q1/datasets/datas"
HEIGHT=224
WIDTH=224
import torch

def preprocessing(path):
    X_img=[]
    y_img=[]
    for imageDir in os.listdir(ImagePath):
        img = cv2.imread(ImagePath + '/' + imageDir)
        img = img.astype(np.float32)
        resized_img_lab = cv2.resize(img, (WIDTH, HEIGHT)) # resize image to network input size
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
        # img = img.astype(np.float32)
        # img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_lab = torch.FloatTensor(np.transpose(color.rgb2lab(np.array(resized_img_lab)), (2, 0, 1)))
                # l_ch_img = resized_img_lab[:,:,0] # pull out L channel
        # img_ab = resized_img_lab[:,:,1:] # Extracting the ab channel
        # img_ab = img_ab / 128 #normalized
        X_img.append(img_lab)
        # y_img.append(img_ab)

    # X_img = np.array(X_img)
    # y_img = np.array(y_img)
    
    return X_img