import struct
import random
import numpy as np
from PIL import Image


MAGIC_NUMBER_LABEL = 2049
MAGIC_NUMBER_IMAGE = 2051


def __load_32_bit__(file):

    return struct.unpack('>L', file.read(4))[0]


def __load_byte__(file):

    return struct.unpack('>B', file.read(1))[0]


def __get_labels__(file, items):

    return np.fromfile(file=file, dtype='uint8', count=items)


def __get_images__(file, items, item_elements):

    all_images = np.zeros(shape=(items, item_elements), dtype='float')
    raw_data = np.fromfile(file=file, dtype='uint8', count=item_elements*items)

    for i in range(0, items):
        all_images[i] = raw_data[i * item_elements: (i + 1) * item_elements]

    return all_images


def __change_to_vector_labels_(labels):

    vector_labels = np.zeros(shape=(labels.size, 10), dtype='uint8')
    index = 0

    for label in labels:
        new_label = np.zeros(10)
        new_label[label] = 1
        vector_labels[index] = new_label
        index += 1

    return vector_labels


def load_label_file(name):

    file = open(name, 'rb')
    x = __load_32_bit__(file)

    if x != MAGIC_NUMBER_LABEL:
        raise Exception('Invalid beginning magic number at file ' + name)

    items = __load_32_bit__(file)
    labels = __get_labels__(file, items)
    labels = __change_to_vector_labels_(labels)
    file.close()

    return labels


def load_images_file(name):

    file = open(name, 'rb')
    x = __load_32_bit__(file)

    if x != MAGIC_NUMBER_IMAGE:
        raise Exception('Invalid beginning magic number at file ' + name)

    images, rows, columns = __load_32_bit__(file), __load_32_bit__(file), __load_32_bit__(file)
    images_array = __get_images__(file, images, rows * columns)
    file.close()

    return images_array / 255


def show_image(image_data):

    size = (512, 512)
    data = image_data * 255
    proper_image = np.reshape(data.astype('uint8'), (28, 28))
    image = Image.fromarray(obj=proper_image, mode='L')
    image = image.resize(size=size)
    image.show()


if __name__ == '__main__':

    TRAIN_IMAGES = 'dataset/train/train-images.idx3-ubyte'
    TRAIN_LABELS = 'dataset/train/train-labels.idx1-ubyte'

    try:

        train_labels, train_images = load_label_file(TRAIN_LABELS), load_images_file(TRAIN_IMAGES)
        print('Train labels size is {}'.format(train_labels.size))
        print('Train labels sample: ' + str(train_labels[0:20]))
        print('Train images shape: {}'.format(train_images.shape))
        print('Showing sample image:')
        show_image(train_images[random.randrange(0, train_labels.shape[0] + 1)])

    except Exception as e:
        print(e)
