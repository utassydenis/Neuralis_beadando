from alap_layer import AlapLayer
import numpy as np

class Retegek(AlapLayer):
    def __init__(self, input_meret, output_meret):
        self.weights = np.random.randn(output_meret, input_meret)
        self.bias = np.random.randn(output_meret, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, dacti_output , learning_rate):
        weights_gradient = np.dot(dacti_output, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * dacti_output
        return np.dot(self.weights.T, dacti_output)

