import torch
import dataloaders
import deeplearning
import nets
import utils
import losses
import gc
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

def test(model, optimizer):
    learning_rate, batch_size, _, dataset_path, _, _, gamma, momentum, model_save_path, model_load_path, cnnBlockType = utils.read_configs()
    test_loader = dataloaders.get_test_dataloader(batch_size, dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    train_accuracy = utils.AverageMeter()
    average_train_loss = utils.AverageMeter()
    val_accuracy = utils.AverageMeter()
    average_val_loss = utils.AverageMeter()
    criterion = losses.cross_entropy()
    with torch.no_grad():
        loop_val = tqdm(
            enumerate(test_loader, 1),
            total=len(test_loader),
            desc="test",
            position=0,
            leave=True,
        )
        for _, (images, labels) in loop_val:
            optimizer.zero_grad()
            images = images.to(device).float()
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            acc1 = val_accuracy.accuracy(labels_pred, labels)
            val_accuracy.update(acc1[0], images.size(0))
            average_val_loss.update(loss.item(), images.size(0))
                
            loop_val.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_val_loss_till_current_batch="{:.4f}".format(average_val_loss.avg),
                validation_accuracy="{:.4f}".format(val_accuracy.avg),
                refresh=True,
            )
            del images, labels, labels_pred
            torch.cuda.empty_cache()
            gc.collect()    
     
        for _ in range(3):
            sample = next(iter(test_loader))
            plot_input_sample(model, batch_data=sample,
                  mean = [0.49139968, 0.48215827 ,0.44653124],
                  std = [0.24703233,0.24348505,0.26158768],
                  to_denormalize = True,
                  figsize = (3,3))


def plot_input_sample(model, batch_data,
                      mean = [0.49139968, 0.48215827 ,0.44653124],
                      std = [0.24703233,0.24348505,0.26158768],
                      to_denormalize = False,
                      figsize = (3,3)):
    
    batch_image, _ = batch_data
    batch_size = batch_image.shape[0]
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
    random_batch_index = random.randint(0,batch_size-1)
    random_image = batch_image[random_batch_index]
    model.to('cpu')
    random_image.to('cpu').float()
    random_image1 = random_image[None, :, :, :]
    predicted = model(random_image1)
    labelIndex = torch.argmax(predicted).int()
    label = classes[labelIndex]
    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    if to_denormalize:
        image_transposed = np.array(std) * image_transposed + np.array(mean)
        image_transposed = image_transposed.clip(0,1)
    _, ax = plt.subplots(1, figsize=figsize)
    plt.suptitle(label)
    ax.imshow(image_transposed)
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    test(None, None)
