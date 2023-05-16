
import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import torch
from torch.utils.data import DataLoader, Dataset
import datasets


def get_train_dataloaders():
    ImagePath = "F:/Master/Deep learning/HW/hw3/Q1/datasets/datas"
    x_data = datasets.preprocessing(ImagePath)
    # train_data = []
    # for i in range(len(x_data)):
        # train_data.append([x_data[i], labels[i]])
    return torch.utils.data.DataLoader(x_data, shuffle=True, batch_size=32)