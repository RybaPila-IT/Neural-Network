from network import VanillaNetwork
from loader import load_images_file, load_label_file

TRAIN_IMAGES = 'dataset/train/train-images.idx3-ubyte'
TRAIN_LABELS = 'dataset/train/train-labels.idx1-ubyte'

def estimate(network, data):
    correct, wrong = 0, 0

    for image, label in data:
        prediction = network.predict(image)

        if (prediction.argmax() == label.argmax()):
            correct += 1
        else:
            wrong += 1

    print('Correct: {} Wrong: {}'.format(correct, wrong))



if __name__ == '__main__':
    train_images = load_images_file(TRAIN_IMAGES)
    train_labels = load_label_file(TRAIN_LABELS)
    network = VanillaNetwork([784, 25, 10])
    #print(network.feed_forward(images[0]))
    # training_set = [(x, y) for x, y in zip(images[0:10], labels[0:10])]
    # network.SGD(training_data=training_set, epochs=2, batch_size=2)
    training_data = [(x, y) for x, y in zip(train_images[0:15000], train_labels[0:15000])]
    validation_data = [(x, y) for x, y in zip(train_images[30001:], train_labels[30001:])]

    network.sgd(training_data, 30, 20, eta=3.5, verbose=True)
    estimate(network, validation_data)




