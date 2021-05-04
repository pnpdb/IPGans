import csv
import glob
import numpy as np
import tensorflow as tf
from setting import config


def celeba_loader():
    try:
        with open(config.CELEBA_PATH + "CelebA_Names.txt", "r") as names:
            true_files = np.array([line.rstrip() for line in names])
            print("CelebA Data File Found. Reading filenames")
    except:
        true_files = sorted(glob.glob(config.CELEBA_PATH + 'img_align_celeba/*.jpg'))
        print("CelebA Data File Created. Saving filenames")
        with open("CelebA_Names.txt", "w") as names:
            for name in true_files:
                names.write(str(name) + '\n')

    train_images = np.expand_dims(np.array(true_files), axis=1)

    attr_file = config.CELEBA_PATH + 'list_attr_celeba.csv'

    with open(attr_file, 'r') as a_f:
        data_iter = csv.reader(a_f, delimiter=',', quotechar='"')
        data = [data for data in data_iter]
    label_array = np.asarray(data)

    return train_images, label_array


def gen_func_celeba():
    train_images, data_array = celeba_loader()

    gender = data_array[1:, 21]
    male = gender == '1'
    male = male.astype('uint8')

    bald_labels = data_array[1:, 5]
    bald = bald_labels == '1'
    bald = bald.astype('uint8')

    hat_labels = data_array[1:, -5]
    hat = hat_labels == '1'
    hat = hat.astype('uint8')

    train_images_male = train_images[np.where(male == 1)]  # 84434
    train_images_female = train_images[np.where(male == 0)]  # 118165

    train_images = np.vstack((train_images_male[:7500], train_images_female[:7500]))

    # train_images = np.random.shuffle(train_images)[:100000]

    # if cls == 'female':
    #     train_images = train_images[np.where(male == 0)]
    # if cls == 'male':
    #     train_images = train_images[np.where(male == 1)]
    # if cls == 'fewfemale':
    #     train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few], 20, axis=0)
    # if cls == 'fewmale':
    #     train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few], 20, axis=0)
    # if cls == 'bald':
    #     train_images = np.repeat(train_images[np.where(bald == 1)], 20, axis=0)
    # if cls == 'hat':
    #     train_images = np.repeat(train_images[np.where(hat == 1)], 20, axis=0)

    return train_images


num_parallel_calls = 4


def dataset_celeba(train_data, buffer_size, batch_size, output_size):
    def data_reader_faces(filename):
        with tf.device('/CPU'):
            image_string = tf.io.read_file(tf.cast(filename[0], dtype=tf.string))
            image = tf.image.decode_jpeg(image_string, channels=3)
            image.set_shape([218, 178, 3])
            image = tf.image.crop_to_bounding_box(image, 38, 18, 140, 140)
            image = tf.image.resize(image, [output_size, output_size])
            image = tf.divide(image, 255.0)

        return image

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
    train_dataset = train_dataset.map(data_reader_faces, num_parallel_calls=int(num_parallel_calls))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(10)

    return train_dataset
