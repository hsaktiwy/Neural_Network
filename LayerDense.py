import numpy as np
import nnfs
import ActivationsFunction as Activation
import Loss

from nnfs.datasets import spiral

nnfs.init()
np.random.seed(0)

# this will represent a layer format

class Layer_Dense:
    def __init__(self, n_input, n_neurons):
        # normaly this matrix dimension will be like (suppose that n_input:4, n_neurones:3) [line :3,colone:4]
        # but we did make it [line: 4, colone: 3] so we don't need to transpose it every time we make matrix dot product
        self.weights=0.10*np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral.create_data(samples=100,classes=3)
#layer1 3 neurons Relu
print("Relu layer1:")
layer1 = Layer_Dense(2,3)
layer1.forward(X)
activation1 = Activation.Relu()
activation1.forward(layer1.output)
print(activation1.output[:5])
print("Softmax layer2:")
#layer2 3 neurons Softmax
layer2 = Layer_Dense(3,3)
layer2.forward(activation1.output)
activation2 = Activation.Softmax()
activation2.forward(layer2.output)
print(activation2.output[:5])
print("Calculating the Loss : ")
loss_function = Loss.CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss :" ,loss)
print("accuracy:", Loss.accuracy(activation2.output, y))