import matplotlib.pyplot as plt
import numpy as np
import sys

# np.savez(train_losses=train_losses,train_top1s=train_top1s,train_top5s=train_top5s, test_losses=test_losses,test_top1s=test_top1s, test_top5s=test_top5s)
def plot_data(filepath):
    data = np.load(filepath)

    if 'train_top1s' in data:
        plt.plot(data['train_top1s'], '-o', label = 'train top1 precision')
        plt.xlabel('epoch Number')
    if 'test_top1s' in data:
        plt.plot(data['test_top1s'], '-o', label = 'validation top1 precision')
        plt.xlabel('epoch Number')
    plt.ylabel('Train/val top1 precision at each epoch')
    plt.legend()
    plt.savefig('Train/val top1 precision at each epoch')
    # plt.show()

    if 'train_top5s' in data:
        plt.plot(data['train_top5s'], '-o', label = 'train top5 precision')
        plt.xlabel('epoch Number')
    if 'test_top5s' in data:
        plt.plot(data['test_top5s'], '-o', label = 'validation top5 precision')
        plt.xlabel('epoch Number')
    plt.ylabel('Train/val top5 precision at each epoch')
    plt.legend()
    plt.savefig('Train/val top1 precision at each epoch')
    # plt.show()

    if 'train_losses' in data:
        plt.plot(data['train_losses'], '-o', label = 'train loss')
        plt.xlabel('epoch Number')
    if 'test_losses' in data:
        plt.plot(data['test_losses'], '-o', label = 'validation loss')
        plt.xlabel('epoch Number')
    plt.ylabel('Train/val loss at each epoch')
    plt.legend()
    plt.savefig('Train/val top1 precision at each epoch')
    # plt.show()



if __name__=='__main__':
    filepath = sys.argv[1]
    plot_data(filepath)