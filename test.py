import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
import torchvision.utils
import cv2


def test(model, optimizer, test_dataset, device):
    with torch.no_grad():
        image_inds = [0, 1]
        lab_batch = torch.stack([test_dataset[i] for i in image_inds])
        # predict colors (ab channels)
        lab_batch = lab_batch.to(device)

        predicted_ab_batch = model(lab_batch[:, 0:1, :, :])
        predicted_lab_batch = torch.cat([lab_batch[:, 0:1, :, :], predicted_ab_batch], dim=1)
        lab_batch = lab_batch.cpu()
        predicted_lab_batch = predicted_lab_batch.cpu()

        # convert to rgb
        rgb_batch = []
        predicted_rgb_batch = []
        for i in range(lab_batch.size(0)):
            rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :], (1, 2, 0)))
            rgb_batch.append(rgb_img)
            predicted_rgb_img = color.lab2rgb(np.transpose(predicted_lab_batch[i, :, :, :], (1, 2, 0)))
            predicted_rgb_img = (255 * predicted_rgb_img).astype("uint8")
            predicted_rgb_batch.append(predicted_rgb_img)
        # plot images
        _, ax = plt.subplots(figsize=(10, 10), nrows=len(image_inds), ncols=2)
        for i in range(len(image_inds)):
            ax[i][0].imshow(predicted_rgb_batch[i])
            ax[i][0].title.set_text('re-colored')
            ax[i][1].imshow(rgb_batch[i])
            ax[i][1].title.set_text('original')
        plt.show()



if __name__ == "__main__":
    test(None, None)

