import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import gc
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import random
from tqdm import tqdm
import losses
import utils


def train(
    train_loader,
    val_loader,
    model,
    epochs,
    learning_rate,
    gamma,
    device,
):
    torch.manual_seed(42)
    model = model.to(device)

    # get loss function from losses module
    criterion = losses.mse_loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = StepLR(optimizer, gamma=gamma,  step_size=2)
    
    val_loss = []
    train_loss = []
    for epoch in tqdm(range(1, epochs + 1)):
        # define evaluation parameters
        average_train_loss = utils.AverageMeter()
        average_val_loss = utils.AverageMeter()
        model.train()
        loop_train = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="train", position=0, leave=True)
        for _, images in loop_train:
            images = images.to(device)
            labels_pred = model(images[:, 0:1, :, :])
            loss = criterion(labels_pred, images[:, 1:3, :, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_train_loss.update(loss.item(), images.size(0))
            # clear gpu cache
            del images, labels_pred
            torch.cuda.empty_cache()
            gc.collect()

            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(average_train_loss.avg),
                refresh=True,
            )
        
        train_loss.append(average_train_loss.avg)
        
        model.eval()
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader, 1), total=len(val_loader), desc="val", position=0, leave=True)
            for _, images in loop_val:
                optimizer.zero_grad()
                images = images.to(device).float()
                labels_pred = model(images[:, 0:1, :, :])
                loss = criterion(labels_pred, images[:, 1:3, :, :])
                average_val_loss.update(loss.item(), images.size(0))
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(average_val_loss.avg),
                    refresh=True
                )
                del images, labels_pred
                torch.cuda.empty_cache()
                gc.collect()

            val_loss.append(average_val_loss.avg)
        lr_scheduler.step()
    return model, optimizer, val_loss, train_loss

