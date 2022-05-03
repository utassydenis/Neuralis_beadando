from activation_reteg import Activacio
import numpy as np

class Tanh(Activacio):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        dacti_tanh = lambda x: 1 - np.tanh(x) ** 2
        super(Tanh, self).__init__(tanh,dacti_tanh)