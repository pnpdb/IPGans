import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualization(real_data, fake_data):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(real_data[0], real_data[1], real_data[2])
    ax.scatter(fake_data[0], fake_data[1], fake_data[2])

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig(time.strftime("%Y-%m-%d %H:%M:%S") + '.jpg')
    plt.show()
