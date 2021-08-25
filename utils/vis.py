import matplotlib.pyplot as plt
import numpy as np

def draw_hist_loss(hist_train_ls, hist_val_ls, dpi=200, save_path=None):
    plt.figure(dpi=dpi)
    plt.plot(
        np.arange(1, len(hist_train_ls)+1), hist_train_ls, 
        '-', label='train loss')
    plt.plot(
        np.arange(1, len(hist_val_ls)+1), hist_val_ls, 
        '-', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def draw_hist_acc(hist_train_acc, hist_val_acc, dpi=200, save_path=None):
    plt.figure(dpi=dpi)
    plt.plot(
        np.arange(1, len(hist_train_acc)+1), hist_train_acc, 
        '-', label='train acc')
    plt.plot(
        np.arange(1, len(hist_val_acc)+1), hist_val_acc, 
        '-', label='validation acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def draw_confusion_matrix(confusion_mat, classes, save_path=None, dpi=200):

    plt.figure(dpi=dpi)

    # Draw the temperature figure.
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    indices = range(len(confusion_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')

    # Print the numbers of records.
    for row in range(len(confusion_mat)):
        for col in range(len(confusion_mat)):
            x, y = col, row
            plt.text(x, y, confusion_mat[row][col])

    if not save_path is None:
        plt.savefig(save_path)
    # Show the figure.
    plt.show()