
import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import torch
from torch.utils.data import DataLoader, Dataset
import datasets


def get_train_dataloaders(dataset_path, batch_size):
    x_data = datasets.preprocessing(dataset_path)
    return torch.utils.data.DataLoader(x_data, shuffle=True, batch_size=batch_size)