import numpy as np

from ActivationsFunction import Tanh
from Loss import MeanSquaredError, CrossEntropyLoss

class NeuralNetwork:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.activation = Tanh()

    def forward(self, inputs):
        return self.activation.forward(inputs)

    def backward(self, dvalues):
        return self.activation.backward(dvalues)

    def calculate_loss(self, y_true, y_pred):
        return self.loss_function.calculate(y_true, y_pred)

    def calculate_loss_gradient(self, y_true, y_pred):
        return self.loss_function.gradient(y_true, y_pred)
