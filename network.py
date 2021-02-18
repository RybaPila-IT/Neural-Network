import numpy as np
import matplotlib.pyplot as plt
import json
import sys


class AccuracyMetric:
    """Class computing accuracy of predicted samples.

    Accuracy is expressed as the ratio: correct_guesses / all_guesses.
    """

    @staticmethod
    def metric_value(x, y):
        """Metric computes ration of correct guesses to all guesses."""
        return (sum(np.all(x == y, axis=1) * 1) / len(x)) * 100


class Sigmoid:
    """Class representing sigmoid activation function.

    Sigmoid class implements popular logistic function called sigmoid function.
    Sigmoid function for argument z is specified as follows:
        1 + / (1 + e ^ (-z))
    """

    @staticmethod
    def activate(z):
        """Sigmoid activation function.

        Sigmoid activation function for arg z is given as:
            1 + / (1 + e ^ (-z))
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        """Sigmoid derivative.

        Sigmoid derivative for arg z is given as:
            sig(z) * (1 - sig(z))   where sig(z) is sigmoid activation function.
        """
        return Sigmoid.activate(z) * (1 - Sigmoid.activate(z))


class SquaredSum:
    """Class representing squared sum cost function.

    Squared sum cost function is given as:
        sum(for all k) [a_k - y] ^ 2    where a_k is the prediction of y value.
    """

    @staticmethod
    def prediction_cost(a, y):
        """Squared sum prediction cost for single activation a.

        Squared sum cost function is given as:
            sum(for all k) [a_k - y_k] ^ 2  where a_k is the prediction of y_k value.
        """
        return np.sum((a - y) ** 2) / 2

    @staticmethod
    def derivative(a, y, z):
        """Squared sum derivative with respect to last neural network layer.

        Squared sum derivative is given as:
            (a - y) * sig_d(z)  where sig_d is sigmoid derivative of last layer weighted input.
        """
        return (a - y) * Sigmoid.derivative(z)


class CrossEntropy:
    """Cross entropy cost function.

    Cross entropy cost function is given as:
        sum(for all k) [ -{y_k * ln(a_k) + (1 - y_k) * ln(1 - a_k)}]    where a_k is prediction of y_k value
    """

    @staticmethod
    def prediction_cost(a, y):
        """Cross entropy prediction cost.

        Cross entropy cost function is given as:
            sum(for all k) [ -{y_k * ln(a_k) + (1 - y_k) * ln(1 - a_k)}]    where a_k is prediction of y_k value
        """
        return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a)))

    @staticmethod
    def derivative(a, y, z):
        """Derivative of cross entropy cost function with respect to last layer weights.

        Cross function derivative is given as:
            a - y   where a is prediction of y
        """
        return a - y + (z * 0)


class Network:
    """Class representing neural network.

    Network class is the most implementation if basic neural network
    using regular stochastic gradient descent as the learning algorithm.
    User is capable of specifying mini-batch size and learning
    rate eta. Those two parameters may have crucial influence
    on network performance while learning.
    Network uses backpropagation algorithm in order to
    compute all necessary partial derivatives further used during cost
    function minimization.
    Moreover Network class supports L2 regularization.
    In order to include regularization while learning one
    should specify lambda_r parameter different than 0.
    """

    def __init__(self, layer_sizes, act_func=Sigmoid, cost_func=CrossEntropy, metric=AccuracyMetric):
        """Constructor of the network.

        cost_func must implement prediction_cost(a, y) and derivative(a, y, z)
        act_func must implement activate(z) and derivative(z)
        metric must implement metric_value(x, y)

        :param layer_sizes - iterable object specifying neural network sizes. Ex. [78, 1] or [784, 80, 10]
        :param act_func - activation function. Default function is sigmoid function.
        :param cost_func - cost function. Default cost function is cross entropy cost function.
        :param metric - metric which will be used while performing accuracy estimation on test set during learning
        """
        np.random.seed(1)  # Used for constant weights and biases initialization. Fell free to change it.

        self.layers_num = len(layer_sizes)
        self.act_func = act_func
        self.cost_func = cost_func
        self.metric = metric
        self.biases = [np.random.random(i) for i in layer_sizes[1:]]
        self.weights = [np.random.normal(loc=0, scale=(1 / np.sqrt(layer_sizes[0])), size=(j, i))
                        for j, i in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.costs = []
        self.accuracies = []
        self.eta = 0
        self.lambda_r = 0

    def save_network(self, file_name):
        """Store network in json file format.

        Function will save network into file in json format.

        :param file_name - name of the file into which network will be saved.
        """
        data = {
            "layers_num": self.layers_num,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "act_func": str(self.act_func.__name__),
            "cost_func": str(self.cost_func.__name__),
            "metric": str(self.metric.__name__)
        }

        with open(file_name, 'w') as file:
            json.dump(data, file)

    def feed_forward(self, a):
        """Front network propagation.

        Feed_forward function performs network activating next layers
        which finally give (in output neurons) results.
        Function returns tuple containing:
        z_arr (weighted inputs of neurons), a_arr (layers activations), a (last layer activation).

        :param a - input neurons activation.

        :returns tuple (z_arr, a_arr, a)
        """
        z_arr, a_arr = [], [a]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w.T) + b  # Next layer weighted input.
            a = self.act_func.activate(z)  # Next layer activation.
            z_arr.append(z), a_arr.append(a)

        return z_arr, a_arr, a

    def predict(self, x):
        """Function predicting output for input x.

        Function, after performing forward propagation in the network,
        will set output with highest value as the predicted output
        (in case of multiple output neurons in last layer).
        """
        return np.array([(i == np.max(i)) * 1 for i in self.feed_forward(x)[2]])

    def plot_costs(self, threshold=0):
        """Function plotting costs of test set after each epoch.

        Function will plot cost of entire training set at each epoch.
        Costs can be restored as long as the next training did not
        take place.
        This graph is useful for monitoring network learning progress.

        :param threshold - information, from which epoch start to display data. Default is 0.
        """
        epochs_range = np.arange(threshold, len(self.costs), 1)
        plt.plot(epochs_range, self.costs[threshold:], color='green', marker='o')
        plt.title('Cost function plot. Eta={:.2f} Lambda={:2.2f}'.format(self.eta, self.lambda_r))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

    def plot_metric_values(self, threshold=0):
        """Function plotting metric values on validation set.

        Function will plot metric value calculated on validation set during
        learning process. Metric values can be restored as long as
        next training did not take place.
        This graph is useful for monitoring overfitting and overall
        network learning progress.

        :param threshold - information, from which epoch start to display data. Default is 0.
        """
        epochs_range = np.arange(threshold, len(self.accuracies), 1)
        plt.plot(epochs_range, self.accuracies[threshold:], color='red', marker='o')
        plt.title('Accuracy on test data. Eta={:.2f} Lambda={:2.2f}'.format(self.eta, self.lambda_r))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    def weights_cost(self, n, lambda_r=0.0):
        """Function calculating cost associated with weights of network.

        Function is being used for including regularization in
        learning process. Weights cost is described as:
            (lambda / 2n) * Sum(for all w in weights) w ^ 2.    where n is the number of training examples.

        :param n - number of examples in training data set.
        :param lambda_r - term lambda associated with the strength of regularization. Default is 0.
        """
        return lambda_r * np.sum(np.sum(w ** 2) for w in self.weights) / (2 * n)

    def backpropagation(self, x, y):
        """Function calculating partial derivatives of cost function for weights and biases.

        Backpropagation is the heart of neural network.
        As the result of performing, it returns tuple of lists
        containing partial derivatives of cost function
        with respect to weights and biases.

        :param x - input data, which is going to be predicted.
        :param y - labels associated with corresponding input data.

        :returns (delta_w, delta_b) where delta_w is list of partial derivatives
                 of cost function with respect to weights and delta_b is
                  list of partial derivatives of cost function with respect to biases.
        """
        result = self.feed_forward(x)
        z_arr, a_arr = result[0], result[1]

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        delta = self.cost_func.derivative(a_arr[-1], y, z_arr[-1])
        delta_b[-1] = np.sum(delta, axis=0)
        delta_w[-1] = np.dot(delta.T, a_arr[-2])

        for layer in range(2, self.layers_num):
            z_d = self.act_func.derivative(z_arr[-layer])
            delta = np.dot(self.weights[-layer + 1].T, np.sum(delta, axis=0)) * z_d
            delta_b[-layer] = np.sum(delta, axis=0)
            delta_w[-layer] = np.dot(delta.T, a_arr[-layer - 1])

        return delta_w, delta_b

    def gradient_descent_step(self, data_batch, eta, lambda_r, n):
        """Step of gradient descent algorithm.

        Function, after computing partial derivatives for ``data_batch``
        will perform weights and biases update.
        Gradient descent step supports regularization.

        :param data_batch - data on which partial derivatives will be evaluated.
        :param eta - learning rate coefficient.
        :param lambda_r - regularization coefficient.
        :param n - size of whole training data.
        """
        images = np.array([x_ for x_, y_ in data_batch])
        labels = np.array([y_ for x_, y_ in data_batch])

        gradient_weights, gradient_biases = self.backpropagation(images, labels)

        self.weights = [(1 - lambda_r * eta / n) * w - (eta / len(data_batch)) * nw
                        for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b - (eta / len(data_batch)) * nb
                       for b, nb in zip(self.biases, gradient_biases)]

    def sgd(self, training_data, epochs, batch_size, eta=0.01, lambda_r=0.2, verbose=False, test_data=None):
        """Stochastic gradient descent.

        Stochastic gradient descent will split ``training data`` into portions
        of size specified as the ``batch_size``. For each batch, there will
        be gradient descent step performed with parameters ``eta`` and ``lambda_r``.

        In order to obtain feedback from learning process specify ``verbose`` as True.
        ``Test_data`` parameter is used as the validation set for measuring metric
        value during learning epochs.

        During learning, if network spots slowdown in learning progress (cost function
        value on test set will not be decreasing) ``eta`` parameter will shrink.

        :param training_data - training data.
        :param epochs - maximal number of learning epochs.
        :param batch_size - size of training batch.
        :param eta - learning rate coefficient.
        :param lambda_r - regularization coefficient.
        :param verbose - whether to give feedback during learning process.
        :param test_data - validation set for estimating metric value.
        """
        stagnation_epochs = 5
        eta_divide_factor = 4
        eta_decrease_available = 4

        self.accuracies = np.zeros(epochs)
        self.costs = np.zeros(epochs)
        self.eta = eta
        self.lambda_r = lambda_r

        n = len(training_data)
        curr_stagnation = stagnation_epochs
        min_cost = float('inf')

        for idx, e in enumerate(range(epochs)):
            np.random.shuffle(training_data)
            batches = [np.array(training_data[k:k + batch_size], dtype=object)
                       for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.gradient_descent_step(batch, eta, lambda_r, n)

            self.costs[idx] = self.cost_func.prediction_cost(self.feed_forward([x_ for x_, y_ in training_data])[2],
                                                             np.array([y_ for x_, y_ in training_data])) + \
                self.weights_cost(n, self.lambda_r)

            if verbose:
                print('Epoch {} ended with cost {:.3f}'.format(e + 1, self.costs[idx]))

            min_cost = min(min_cost, self.costs[idx])

            if idx != 0:
                curr_stagnation = curr_stagnation - 1 if self.costs[idx] != min_cost else stagnation_epochs

            if curr_stagnation == 0:
                curr_stagnation = stagnation_epochs
                eta /= eta_divide_factor
                eta_decrease_available -= 1
                min_cost = float('inf')

                if eta_decrease_available < 0:
                    if verbose:
                        print('Eta decreased maximum available times, terminating')
                    break

                if verbose:
                    print('Decreasing eta to {:.5f}'.format(eta))

            if test_data:
                self.accuracies[idx] = self.metric.metric_value(self.predict([x_ for x_, y_ in test_data]),
                                                                [y_ for x_, y_ in test_data])

        if verbose:
            self.plot_costs()

            if test_data:
                self.plot_metric_values()


def load_network(file_name):
    """Function will load network from file.

    File_name should bo a file containing network parameters
    specified as the json file format.

    :param file_name - json file name containing network parameters.
    :returns network with pre-learned parameters.
    """
    with open(file_name) as file:
        data = json.load(file)

    cost_fn = getattr(sys.modules[__name__], data["cost_func"])
    act_fn = getattr(sys.modules[__name__], data["act_func"])
    metric = getattr(sys.modules[__name__], data["metric"])

    network = Network([1, 1], act_func=act_fn, cost_func=cost_fn, metric=metric)
    network.layers_num = data["layers_num"]
    network.weights = [np.array(w) for w in data["weights"]]
    network.biases = [np.array(b) for b in data["biases"]]

    return network
