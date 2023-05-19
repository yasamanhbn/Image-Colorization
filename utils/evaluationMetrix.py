from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss):
  _, axes = plt.subplots(1, 1, figsize=(12, 3))
  plt.suptitle("Loss plot for train and test data", size=14)
  axes.plot(list(range(1, len(train_loss) + 1)), train_loss, 'b',label='train loss')
  axes.plot(list(range(1, len(test_loss) + 1)), test_loss, 'r', label='test loss')
  axes.set_ylabel('Loss Average', size=12, labelpad=10)
  axes.set_xlabel('Epoch', size=12, labelpad=10)
  axes.legend(loc='lower right', fontsize=10)
  axes.grid()
  plt.show()