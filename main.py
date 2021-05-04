import tensorflow as tf
from models.celeba.iSDCGans import IDCGans

if __name__ == "__main__":
    # tf.enable_eager_execution()
    s_dcgan = IDCGans(epochs=100, start=0)
    s_dcgan.train()
