import numpy as np

class Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

class Sigmoid:
    def forward(self, inputs):
        self.output= 1 / (1 + np.exp(inputs))

class Tanh:
    def forward(self, inputs):
        e_ = np.exp(-inputs)
        e = np.exp(inputs)
        self.output= (e - e_) /(e + e_)