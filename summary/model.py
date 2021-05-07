import tensorflow as tf
from tensorflow.keras import layers

init_bute = tf.keras.initializers.glorot_uniform()
initializer = tf.function(init_bute, autograph=False)


def make_discriminator_model(output_size=28):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, input_shape=[output_size, output_size, 3]))  # 32x32x64
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer))  # 16x16x128
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer))  # 8x8x256
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer))  # 4x4x512
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())  # 8192x1
    model.add(layers.Dense(1))  # 1x1

    return model


def make_generator_model(output_size=28):
    inputs = tf.keras.Input(shape=(100,))

    dec1 = tf.keras.layers.Dense(int(output_size / 16) * int(output_size / 16) * 1024, kernel_initializer=initializer, use_bias=False)(inputs)
    dec1 = tf.keras.layers.LeakyReLU()(dec1)

    un_flat = tf.keras.layers.Reshape([int(output_size / 16), int(output_size / 16), 1024])(dec1)  # 4x4x1024

    deconv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)(un_flat)  # 8x8x512 , New is 512
    deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
    deconv1 = tf.keras.layers.LeakyReLU()(deconv1)

    deconv2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)(
        deconv1)  # 16x16x256 , New is 512
    deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
    deconv2 = tf.keras.layers.LeakyReLU()(deconv2)

    deconv4 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)(
        deconv2)  # 32x32x128 , New is 1024
    deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
    deconv4 = tf.keras.layers.LeakyReLU()(deconv4)

    out = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer, activation='sigmoid')(
        deconv4)  # 64x64x3

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


d = make_discriminator_model()
d.summary()
