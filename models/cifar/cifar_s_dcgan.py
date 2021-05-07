import os
import numpy as np
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from models.celeba import gen_func_celeba
from models.celeba import dataset_celeba
from models import config


class CIFARSDCGans:
    def __init__(self, noise_dim=100, output_size=32, epochs=100, start=0):
        print("Creating DCGan Model for MNIST...")
        self.noise_dim = noise_dim
        self.output_size = output_size
        self.epochs = epochs
        self.start = start
        self.BUFFER_SIZE = 2048
        self.BATCH_SIZE = 256
        self.num_exm_images = 16
        self.noise_mean = 0
        self.noise_stddev = 1.0
        # self.seed = self.load_seed(self.noise_dim)
        self.seed = tf.random.normal([self.num_exm_images, noise_dim], mean=self.noise_mean, stddev=self.noise_stddev)
        self.fid_model = None
        self.init_bute = tf.keras.initializers.glorot_uniform()
        self.initializer = tf.function(self.init_bute, autograph=False)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.g_checkpoint = tf.train.Checkpoint(myModel=self.generator)
        self.d_checkpoint = tf.train.Checkpoint(myModel=self.discriminator)
        self.ck_g_manager = tf.train.CheckpointManager(self.g_checkpoint, directory=config.DCGAN_S_MNIST_CHECKPOINT_G_PATH, checkpoint_name='model.ckpt', max_to_keep=3)
        self.ck_d_manager = tf.train.CheckpointManager(self.d_checkpoint, directory=config.DCGAN_S_MNIST_CHECKPOINT_D_PATH, checkpoint_name='model.ckpt', max_to_keep=3)
        self.fid_scores = []
        self.losses = []
        # self.restore_model()

    def load_seed(self, noise_dim):
        if os.path.exists('seed.npy'):
            print('load seed..........')
            np_arr = np.load('seed.npy')
            return tf.convert_to_tensor(np_arr)
        else:
            tf_arr = tf.random.normal([self.num_exm_images, noise_dim], mean=self.noise_mean, stddev=self.noise_stddev)
            x = tf_arr.cpu().numpy()
            np.save('seed.npy', x)
            return tf_arr

    def make_generator_model(self):
        inputs = tf.keras.Input(shape=(self.noise_dim,))

        enc_res = tf.keras.layers.Reshape([1, 1, int(self.noise_dim)])(inputs)

        denc4 = tf.keras.layers.Conv2DTranspose(512, 5, strides=2, padding='same', kernel_initializer=self.initializer, use_bias=True)(enc_res)  # 2x2x128
        denc4 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc4)
        # denc4 = tf.keras.layers.Dropout(0.5)(denc4)
        denc4 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc4)

        denc3 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same', kernel_initializer=self.initializer, use_bias=True)(denc4)  # 4x4x256
        denc3 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc3)
        # denc3 = tf.keras.layers.Dropout(0.5)(denc3)
        denc3 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc3)

        denc2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same', kernel_initializer=self.initializer, use_bias=True)(denc3)  # 8x8x128
        denc2 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc2)
        # denc2 = tf.keras.layers.Dropout(0.5)(denc2)
        denc2 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc2)

        denc1 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', kernel_initializer=self.initializer, use_bias=True)(denc2)  # 16x16x64
        denc1 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc1)
        # denc1 = tf.keras.layers.Dropout(0.5)(denc1)
        denc1 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc1)

        out = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same', kernel_initializer=self.initializer)(denc1)  # 32x32x3
        out = tf.keras.layers.Activation(activation='tanh')(out)

        model = tf.keras.Model(inputs=inputs, outputs=out)
        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()  # 64x64x3
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=self.initializer, input_shape=[self.output_size, self.output_size, 3]))  # 32x32x64
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.1))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=self.initializer))  # 16x16x128
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.1))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=self.initializer))  # 8x8x256
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.1))

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=self.initializer))  # 4x4x512
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.LeakyReLU(alpha=0.1))

        model.add(layers.Flatten())  # 8192x1
        model.add(layers.Dense(512))
        model.add(layers.Dense(1))  # 1x1

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # calculate frechet inception distance
    @staticmethod
    def calculate_fid(act1, act2):
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate summary squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid

    def load_fid_model(self):
        self.fid_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet',
                                                                        input_tensor=None, input_shape=(80, 80, 3), classes=1000)

    def mnist_fid_score(self, epoch, fid_train_images_names):
        def data_preprocess(image):
            with tf.device(config.DEVICE):
                image = tf.image.resize(image, [80, 80])
            return image

        self.load_fid_model()

        fid_num_samples = 1000
        fid_batch_size = tf.constant(100, dtype='int64')
        num_parallel_calls = 4

        random_points = tf.keras.backend.random_uniform([min(fid_train_images_names.shape[0], fid_num_samples)], minval=0, maxval=int(fid_train_images_names.shape[0]),
                                                        dtype='int32',
                                                        seed=None)

        fid_train_images_names = fid_train_images_names[random_points]

        fid_image_dataset = tf.data.Dataset.from_tensor_slices(fid_train_images_names)
        fid_image_dataset = fid_image_dataset.map(data_preprocess, num_parallel_calls=int(num_parallel_calls))
        fid_image_dataset = fid_image_dataset.batch(fid_batch_size)

        with tf.device(config.DEVICE):
            count = 0
            fid_sum = 0
            for image_batch in fid_image_dataset:
                noise = tf.random.normal([fid_batch_size, self.noise_dim], self.noise_mean, self.noise_stddev)
                preds = self.generator(noise, training=False)
                preds = tf.image.resize(preds, [80, 80])
                preds = tf.scalar_mul(2., preds)
                preds = tf.subtract(preds, 1.0)
                preds = preds.numpy()

                act1 = self.fid_model.predict(image_batch)
                act2 = self.fid_model.predict(preds)
                try:
                    act1 = np.concatenate((act1, act1), axis=0)
                    act2 = np.concatenate((act2, act2), axis=0)
                    fid_score = self.calculate_fid(act1, act2)
                    fid_sum += fid_score
                    count += 1
                except:
                    act1 = act1
                    act2 = act2

            avg_fid_score = fid_sum / count / fid_batch_size
            self.fid_scores.append(str(epoch) + ',' + str(np.array(avg_fid_score)))
            print("epoch: %d fid: %f" % (epoch, avg_fid_score))

    def generate_and_save_images(self, epoch):
        predictions = self.generator(self.seed, training=False)
        # predictions = tf.multiply(predictions, 255.0)
        predictions = predictions.numpy()

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')

        plt.savefig(config.DCGAN_S_MNIST_IMAGE_FORMAT.format(epoch))
        plt.show()

    def save_fid_scores(self):
        with open(config.DCGAN_I_MNIST_FID_FILE, 'w') as f:
            for i in self.fid_scores:
                f.write(str(i) + '\n')

    def save_losses(self):
        with open(config.DCGAN_I_MNIST_LOSS_FILE, 'w') as f:
            for i in self.losses:
                f.write(str(i) + '\n')

    def checkpoint(self, epoch):
        # self.g_checkpoint.save(config.DCGAN_MNIST_CHECKPOINT_G_PATH + '/model.ckpt')
        # self.d_checkpoint.save(config.DCGAN_MNIST_CHECKPOINT_D_PATH + '/model.ckpt')
        self.ck_g_manager.save(checkpoint_number=epoch)
        self.ck_d_manager.save(checkpoint_number=epoch)

    def restore_model(self):
        self.g_checkpoint.restore(tf.train.latest_checkpoint(config.DCGAN_S_MNIST_CHECKPOINT_G_PATH))
        self.d_checkpoint.restore(tf.train.latest_checkpoint(config.DCGAN_S_MNIST_CHECKPOINT_D_PATH))
        print('Restore model from checkpoint......')

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_images = train_images.reshape(train_images.shape[0], self.output_size, self.output_size, 3).astype('float32')
        train_images = (train_images - 127.5) / 127.5

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        for epoch in range(self.start, self.epochs, 1):
            gl, dl = .0, .0
            for i, image_batch in enumerate(train_dataset):
                g, d = self.train_step(image_batch)
                gl, dl = g.numpy(), d.numpy()
                print("epoch: %d, batch: %d, gen_loss: %f, disc_loss: %f" % (epoch, i, g.numpy(), d.numpy()))

            self.losses.append(str(epoch) + ',' + str(gl) + ',' + str(dl))
            self.generate_and_save_images(epoch)
            self.mnist_fid_score(epoch, train_images)
            self.save_fid_scores()
            self.save_losses()
            self.checkpoint(epoch)
            print('epoch: %d done' % epoch)
