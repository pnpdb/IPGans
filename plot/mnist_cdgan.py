import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def avg_fids(f):
    sum = .0
    count = 0
    with open(f, 'r') as f:
        lines = f.readlines()
        count = len(lines)
        for line in lines:
            arr = line.split(',')
            sum += float(arr[1])

    return sum / count


def make_mnist_cdgan_fid():
    x = []
    y1 = []
    y2 = []

    with open('./fid_mnist_s.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            if i % 30 == 0:
                arr = line.split(',')
                x.append(i)
                y1.append(float(arr[1]))

    print(len(x))

    with open('./fid_mnist_i.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            if i % 30 == 0:
                arr = line.split(',')
                y2.append(float(arr[1]))

    plt.plot(x, y1, label='Standard CDGan')
    plt.plot(x, y2, label='Improved CDGan')
    plt.xlabel('epoch')
    plt.ylabel('FID')
    plt.title('FASHION MNIST')  # Fréchet Inception Distance
    plt.legend(loc='upper right', fontsize=12)
    plt.show()


def make_celeba_cdgan_fid():
    x = []
    y1 = []
    y2 = []

    with open('./celeba_s.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            if i % 30 == 0:
                arr = line.split(',')
                x.append(i)
                y1.append(float(arr[1]))

    print(len(x))

    with open('./celeba_i.txt', 'r') as f:
        lines = f.readlines()

        print(len(lines))
        for i, line in enumerate(lines):
            if i % 30 == 0:
                arr = line.split(',')
                y2.append(float(arr[1]))

    plt.plot(x, y1, label='Standard CDGan')
    plt.plot(x, y2, label='Improved CDGan')
    plt.xlabel('epoch')
    plt.ylabel('FID')
    plt.title('CelebA')  # Fréchet Inception Distance
    plt.legend(loc='upper right', fontsize=12)
    plt.show()


make_mnist_cdgan_fid()
make_celeba_cdgan_fid()

a = avg_fids('./celeba_s.txt')
b = avg_fids('./celeba_i.txt')
c = avg_fids('./fid_mnist_s.txt')
d = avg_fids('./fid_mnist_i.txt')
print(a, b, c, d)
