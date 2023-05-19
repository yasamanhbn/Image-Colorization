
import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import torch
from torch.utils.data import DataLoader, Dataset
import datasets
from sklearn.model_selection import train_test_split


def get_dataloaders(dataset_path, batch_size, train_split):
    x_data = datasets.preprocessing(dataset_path)
    train, test = train_test_split(x_data, test_size=train_split, random_state=42)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader, test