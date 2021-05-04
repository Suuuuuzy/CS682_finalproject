import matplotlib.pyplot as plt
import numpy as np
import sys


# np.savez(train_perc_store=train_perc_store,  loss_batch_store=loss_batch_store, test_perc_store=test_perc_store, test_loss_store=test_loss_store)
def plot_data(filepath):
    data = np.load(filepath)

    if 'train_perc' in data:
        plt.plot(data['train_perc'], '-o', label = 'train precision')
        plt.xlabel('epoch Number')
    if 'test_perc' in data:
        plt.plot(data['test_perc'], '-o', label = 'validation precision')
        plt.xlabel('epoch Number')
    plt.ylabel('train/val precision at each epoch')
    plt.legend()
    plt.show()

    if 'train_loss' in data:
        plt.plot(data['train_loss'], '-o', label = 'train loss')
        plt.xlabel('epoch Number')
    if 'test_loss' in data:
        plt.plot(data['test_loss'], '-o', label = 'validation loss')
        plt.xlabel('epoch Number')
    plt.ylabel('train/val loss at each epoch')
    plt.legend()
    plt.show()



if __name__=='__main__':
    filepath = sys.argv[1]
    plot_data(filepath)