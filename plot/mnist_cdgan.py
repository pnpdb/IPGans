import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def make_scdgan_loss():
    x = []
    y_gen = []
    y_dis = []
    with open('./losses.txt', 'r') as f:
        lines = f.readlines()
        print(len(lines))
        for i, line in enumerate(lines):
            arr = line.split(',')
            x.append(i)
            y_gen.append(float(arr[2]))
            y_dis.append(float(arr[3]))

    plt.plot(x, y_gen, label='gen loss')
    plt.plot(x, y_dis, label='dis loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST sCDGAN Loss')
    plt.legend(loc='upper right', fontsize=12)  # lower
    plt.show()


def make_mnist_cdgan_fid():
    x = []
    y1 = []
    y2 = []

    with open('./fids2.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            arr = line.split(',')
            x.append(i)
            y2.append(float(arr[1]))

    with open('./fids.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            arr = line.split(',')
            # x.append(i)
            if i < len(x):
                y1.append(float(arr[1]))

    # plt.plot(x, y, '.-')
    plt.plot(x, y1, label='CDGan')
    plt.plot(x, y2, label='Improved CDGan')
    plt.xlabel('epoch')
    plt.ylabel('distance')
    plt.title('MNIST FID')  # Fréchet Inception Distance
    # plt.fill_between(x, y)
    plt.legend(loc='upper right', fontsize=12)  # lower
    plt.show()


make_mnist_cdgan_fid()
