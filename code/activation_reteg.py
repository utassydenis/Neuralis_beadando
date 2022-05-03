import numpy as np

from alap_layer import AlapLayer

class Activacio(AlapLayer):
    def __init__(self, acti, dacti):
        self.acti = acti
        self.dacti = dacti

    def forward(self, input):
        self.input = input
        return self.acti(self.input)

    def backward(self, dacti_output, learning_rate):
        return np.multiply(dacti_output, self.dacti(self.input))