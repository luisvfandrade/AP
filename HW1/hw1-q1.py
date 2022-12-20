#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        lr = 1
        y_hat = np.argmax(self.W.dot(x_i))
        
        if y_hat != y_i:
            self.W[y_i, :] += lr * x_i
            self.W[y_hat, :] -= lr * x_i
        
        
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Label scores according to the model (num_labels x 1).
        label_scores = self.W.dot(x_i)[:, None]
        
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, hidden_layers):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden = hidden_size
        self.weights = []
        self.biases = []
        for i in range(hidden_layers + 1):
            if i == 0:
                self.weights.append(np.random.normal(0.1, 0.1, (hidden_size, n_features)))
                self.biases.append(np.zeros(hidden_size))
            elif i == hidden_layers:
                self.weights.append(np.random.normal(0.1, 0.1, (n_classes, hidden_size)))
                self.biases.append(np.zeros(n_classes))
            else:
                self.weights.append(np.random.normal(0.1, 0.1, (hidden_size, hidden_size)))
                self.biases.append(np.zeros(hidden_size))

    def one_hot(self, y):
        sample_size = y.shape[0]
        y_one_hot = np.zeros((sample_size, self.n_classes))
        y_one_hot[np.arange(sample_size), y] = 1
        return y_one_hot

    def activation(self, x):
        return x * (x > 0) # ReLU

    def activation_derivative(self, x):
        return np.where(x > 0, 1, 0) # ReLU derivative

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        num_layers = len(self.weights)
        hiddens = []
        for i in range(num_layers):
            h = X if i == 0 else hiddens[i - 1]
            z = self.weights[i].dot(h) + self.biases[i]
            if i < 1:
                hiddens.append(self.activation(z))
        output = z
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat, output, hiddens

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        sample_size = X.shape[0]
        y_one_hot = self.one_hot(y)
        n_correct = 0
        for i in range(sample_size):
            y_hat, _, _ = self.predict(X[i])
            n_correct += 1 if np.array_equal(y_one_hot[i], y_hat) else 0
        return n_correct / sample_size

    def train_epoch(self, X, y, learning_rate=0.001):
        num_layers = len(self.weights)
        sample_size = X.shape[0]
        y_one_hot = self.one_hot(y)
        for i in range(sample_size):
            _, z, hiddens = self.predict(X[i])
            z -= np.max(z)
            probs = np.exp(z) / np.sum(np.exp(z))
            grad_z = probs - y_one_hot[i]
            grad_weights = []
            grad_biases = []
            for j in range(num_layers - 1, -1, -1):
                # Gradient of hidden parameters.
                h = X[i] if j == 0 else hiddens[j - 1]
                grad_weights.append(grad_z[:, None].dot(h[:, None].T))
                grad_biases.append(grad_z)

                # Gradient of hidden layer below.
                grad_h = self.weights[j].T.dot(grad_z)

                # Gradient of hidden layer below before activation.
                derivative = self.activation_derivative(h)
                grad_z = grad_h * derivative  # Grad of loss wrt z3.
            grad_weights.reverse()
            grad_biases.reverse()
            for j in range(num_layers):
                self.weights[j] -= learning_rate*grad_weights[j]
                self.biases[j] -= learning_rate*grad_biases[j]


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    sample_size = train_X.shape[0]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(sample_size)
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
