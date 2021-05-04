# coding:utf-8
class Config:
    DEBUG = True

    def __getitem__(self, item):
        return self.__getattribute__(item)


class LocalConfig(Config):
    DEBUG = True
    DEVICE = '/cpu'
    # standard dcgan
    DCGAN_S_MNIST_IMAGE_FORMAT = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/images/dcgan/s/image_at_epoch_{:04d}.png'
    DCGAN_S_MNIST_FID_FILE = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/s/fids.txt'
    DCGAN_S_MNIST_LOSS_FILE = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/s/losses.txt'
    DCGAN_S_MNIST_CHECKPOINT_G_PATH = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/s/ckpt/g'
    DCGAN_S_MNIST_CHECKPOINT_D_PATH = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/s/ckpt/d'

    # improved dcgan
    DCGAN_I_MNIST_IMAGE_FORMAT = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/images/dcgan/i/image_at_epoch_{:04d}.png'
    DCGAN_I_MNIST_FID_FILE = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/i/fids.txt'
    DCGAN_I_MNIST_LOSS_FILE = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/i/losses.txt'
    DCGAN_I_MNIST_CHECKPOINT_G_PATH = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/i/ckpt/g'
    DCGAN_I_MNIST_CHECKPOINT_D_PATH = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/resources/dcgan/i/ckpt/d'

    CELEBA_PATH = '/Users/lianhai/Resources/illinois/CS584/programming/IPGans/data/CelebA/'


class NvidiaConfig(Config):
    DEBUG = False
    DEVICE = '/gpu:0'

    DCGAN_S_MNIST_IMAGE_FORMAT = '/root/workspace/images/image_at_epoch_{:04d}.png'
    DCGAN_S_MNIST_FID_FILE = '/root/workspace/score/fids.txt'
    DCGAN_S_MNIST_LOSS_FILE = '/root/workspace/score/losses.txt'
    DCGAN_S_MNIST_CHECKPOINT_G_PATH = '/root/workspace/ckpt/g'
    DCGAN_S_MNIST_CHECKPOINT_D_PATH = '/root/workspace/ckpt/d'

    DCGAN_I_MNIST_IMAGE_FORMAT = '/root/workspace/images/image_at_epoch_{:04d}.png'
    DCGAN_I_MNIST_FID_FILE = '/root/workspace/score/fids.txt'
    DCGAN_I_MNIST_LOSS_FILE = '/root/workspace/score/losses.txt'
    DCGAN_I_MNIST_CHECKPOINT_G_PATH = '/root/workspace/ckpt/g'
    DCGAN_I_MNIST_CHECKPOINT_D_PATH = '/root/workspace/ckpt/d'

    CELEBA_PATH = '/root/IPGans/data/CelebA/'


env_config = {
    'dev': LocalConfig,
    'nvidia': NvidiaConfig
}

config = env_config['dev']()
