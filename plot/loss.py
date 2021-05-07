import matplotlib.pyplot as plt


def make_cdgan_loss():
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


make_cdgan_loss()
