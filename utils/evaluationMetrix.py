from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_accuracy(prediction, yTrue):
  assert(len(yTrue) == len(prediction))
  return np.mean(prediction==yTrue)


def get_evaluation_matrics(y_test, y_prediction):
    model_accuracy = get_accuracy(prediction=y_prediction, yTrue=y_test)
    print("Accuracy: {0:.3f}".format(model_accuracy * 100))
    
    recall_test = recall_score(y_test, y_prediction, average='micro')
    print("recall: {0:.3f}".format(recall_test * 100))
    
    precision_test = precision_score(y_test, y_prediction, average='micro')
    print("precision: {0:.3f}".format(precision_test * 100))
    
    f1_test = f1_score(y_test, y_prediction, average='micro')
    print("F1 score: {0:.3f}".format(f1_test * 100))


def plot_acc_loss(train_acc, train_loss, test_acc, test_loss):
  _, axes = plt.subplots(1, 2, figsize=(12, 3))
  plt.suptitle("Accuracy and Loss plot for train and validation data", size=14)
  axes[0].plot(list(range(1, len(train_acc)+1)), train_acc, 'b',label='train accuracy')
  axes[0].plot(list(range(1, len(train_acc)+1)), test_acc, 'r', label='test accuracy')
#   axes[0].set_xlim(1)
  axes[0].set_ylabel('Accuracy Average', size=12, labelpad=10)
  axes[0].set_xlabel('Epoch', size=12, labelpad=10)
  axes[0].legend(loc='lower right', fontsize=10)
  axes[0].grid()

  axes[1].plot(list(range(1, len(train_acc)+1)), train_loss, 'b', label='train errors')
  axes[1].plot(list(range(1, len(train_acc)+1)), test_loss, 'r',label='test errors')
#   axes[1].set_xlim(1)
  axes[1].set_ylabel('Loss Average', size=12, labelpad=11)
  axes[1].set_xlabel('Epoch', size=12, labelpad=10)
  axes[1].legend(loc='best', fontsize=10)
  axes[1].grid()

  plt.show()