import sys
import argparse
import tensorflow as tf
from models.mnist import mnist_s_dcgan, mnist_i_dcgan
from models.celeba import celeba_s_dcgan, celeba_i_dcgan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', required=True, type=str)
    parser.add_argument('--d', required=True, type=str)
    # args = parser.parse_args()

    flags, unparsed = parser.parse_known_args(sys.argv[1:])
    m, d = flags.m, flags.d
    # tf.enable_eager_execution()
    if m == 'standard':
        if d == 'mnist':
            gan = mnist_s_dcgan.MNISTSDCGans(epochs=10000, start=0)
            gan.train()
            # gan.visualize()
        elif d == 'celeba':
            gan = celeba_s_dcgan.CelebASDCGans(epochs=10000, start=0)
            gan.train()

    elif m == 'improved':
        if d == 'mnist':
            gan = mnist_i_dcgan.MNISTIDCGans(epochs=10000, start=0)
            gan.train()
        elif d == 'celeba':
            gan = celeba_i_dcgan.CelebAIDCGans(epochs=10000, start=0)
            gan.train()


if __name__ == '__main__':
    main()
