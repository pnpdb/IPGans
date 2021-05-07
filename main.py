import tensorflow as tf
from models.mnist.iSDCGans import IDCGans

if __name__ == "__main__":
    # tf.enable_eager_execution()
    gan = IDCGans(epochs=100, start=0)
    # gan.train()
    gan.visualize()
