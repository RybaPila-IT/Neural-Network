from network import Network, AccuracyMetric     # , load_network
from loader import load_images_file, load_label_file

TRAIN_IMAGES = 'dataset/train/train-images.idx3-ubyte'
TRAIN_LABELS = 'dataset/train/train-labels.idx1-ubyte'

TEST_IMAGES = 'dataset/test/t10k-images.idx3-ubyte'
TEST_LABELS = 'dataset/test/t10k-labels.idx1-ubyte'


def split_data(data_im, data_l, threshold=50000):
    train_set = [(x, y) for x, y in zip(data_im[0:threshold], data_l[0:threshold])]
    valid_set = [(x, y) for x, y in zip(data_im[threshold:], data_l[threshold:])]

    return train_set, valid_set


if __name__ == '__main__':

    all_train_images = load_images_file(TRAIN_IMAGES)
    all_train_labels = load_label_file(TRAIN_LABELS)

    test_images = load_images_file(TEST_IMAGES)
    test_labels = load_label_file(TEST_LABELS)

    test_set = [(x, y) for x, y in zip(test_images, test_labels)]

    training_set, validation_set = split_data(all_train_images, all_train_labels)

    # Uncomment in order to load network from file.
    # file_name = 'net.json'
    # network = load_network('net.json')

    # Comment in order to omit training.
    network = Network([784, 100, 30, 10])
    network.sgd(training_set, 150, 15, eta=0.15, lambda_r=5, verbose=True, test_data=validation_set)

    print('Accuracy on test set is: {:.2f}%'.format(
        AccuracyMetric.metric_value(network.predict(test_images), test_labels)))

    # Uncomment in order to save training result
    # or specify own file name different than proposed
    # file_name = 'my_net.json'
    # network.save_network(file_name)
