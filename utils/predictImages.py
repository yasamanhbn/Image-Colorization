import os
import glob
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import torch
import dataloaders
import deeplearning
import nets
import utils

learning_rate, batch_size, num_epochs, dataset_path, num_classes, train_split, gamma, momentum, model_save_path, model_load_path, cnnBlockType = utils.read_configs()
train_loader, val_loader = dataloaders.get_train_dataloaders(batch_size)

def plot_input_sample(batch_data,
                      mean = [0.49139968, 0.48215827 ,0.44653124],
                      std = [0.24703233,0.24348505,0.26158768],
                      to_denormalize = False,
                      figsize = (3,3)):
    
    batch_image, _ = batch_data
    batch_size = batch_image.shape[0]
    
    random_batch_index = random.randint(0,batch_size-1)
    random_image = batch_image[random_batch_index]
    
    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    if to_denormalize:
        image_transposed = np.array(std) * image_transposed + np.array(mean)
        image_transposed = image_transposed.clip(0,1)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_transposed)
    ax.set_axis_off()
    plt.show()
    
    
     
sample = next(iter(val_loader))

plot_input_sample(batch_data=sample,
                  mean = [0.49139968, 0.48215827 ,0.44653124],
                  std = [0.24703233,0.24348505,0.26158768],
                  to_denormalize = True,
                  figsize = (3,3))


