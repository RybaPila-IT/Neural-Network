import numpy as np
import matplotlib.pyplot as plt


class VanillaNetwork:
    """Class representing plain neural network

    Vanilla network is the most basic neural network using regular
    stochastic gradient descent as the learning algorithm.
    User is capable of specifying mini-batch size and learning
    rate eta. Those two parameters may have crucial influence
    on network performance while learning.
    Vanilla network uses backpropagation algorithm in order to
    compute all necessary partial derivatives used during cost
    function minimization.
    """

    def __init__(self, layer_sizes, reg_term=1):

        np.random.seed(1)

        self.layers_num = len(layer_sizes)
        self.reg_term = reg_term
        self.biases = [np.random.random(i) for i in layer_sizes[1:]]
        self.weights = [np.random.random(size=(j, i)) * 0.12
                        for j, i in zip(layer_sizes[1:], layer_sizes[:-1])]

    def feed_forward(self, a):

        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def predict(self, x):
        prediction = self.feed_forward(x)
        return np.array([(i == np.max(prediction)) * 1 for i in prediction])

    @staticmethod
    def cost_function_derivative(a, y):
        return a - y

    def backpropagation(self, x, y):
        activation = x
        z_arr, a_arr = [], [activation]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            z_arr.append(z), a_arr.append(activation)

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        delta = self.cost_function_derivative(a_arr[-1], y) * sigmoid_derivative(z_arr[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(np.array([delta]).T, np.array([a_arr[-2]]))

        for layer in range(2, self.layers_num):
            z_d = sigmoid_derivative(z_arr[-layer])
            delta = np.dot(self.weights[-layer + 1].T, delta) * z_d
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(np.array([delta]).T, np.array([a_arr[-layer - 1]]))

        return delta_w, delta_b

    def gradient_descent_step(self, data_batch, eta):
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        cost = 0

        for x, y in data_batch:
            cost += prediction_cost(self.feed_forward(x), y)
            n_weights, n_biases = self.backpropagation(x, y)
            gradient_biases = [gb + nb for gb, nb in zip(gradient_biases, n_biases)]
            gradient_weights = [gw + nw for gw, nw in zip(gradient_weights, n_weights)]

        self.weights = [w - (eta / len(data_batch)) * nw
                        for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b - (eta / len(data_batch)) * nb
                       for b, nb in zip(self.biases, gradient_biases)]

        return cost

    def sgd(self, training_data, epochs, batch_size, eta=0.01, verbose=False):

        costs = np.zeros(epochs)

        for idx, e in enumerate(range(epochs)):
            np.random.shuffle(training_data)
            batches = [np.array(training_data[k:k + batch_size], dtype=object)
                       for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                costs[idx] += self.gradient_descent_step(batch, eta)

            if verbose:
                print('Epoch {} ended with cost {:.3f}'.format(e + 1, costs[idx]))

        if verbose:
            epochs_range = np.arange(0, epochs, 1)
            plt.plot(epochs_range, costs, color='green')
            plt.title('Cost function plot. Eta={:f}'.format(eta))
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def prediction_cost(p, y):
    return np.sum((p - y) ** 2)
