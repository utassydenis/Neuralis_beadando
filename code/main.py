import numpy as np
from retegek import Retegek
from activaciok import Tanh

#Adatfeldolgozás

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


adatok = [sor.strip().split(sep=";") for sor in open("Files/Hungary.csv")]

adatok = [[float(adat) if is_number(adat) else adat  for adat in sor[4:5]] for sor in adatok[1:]]

train = adatok[:500]
test = adatok[500:]

train_x = (np.array([train[i:i+5] for i in range(len(train)-5)])-5)/45050
train_y = (np.array([train[i+5] for i in range(len(train)-5)])-5)/45050

test_x = (np.array([test[i:i+5] for i in range(len(test)-5)])-5)/45050
test_y = (np.array([test[i+5] for i in range(len(test)-5)])-5)/45050

# Neurálisháló
def mse(elvart_output, output):
    return np.mean(np.power(elvart_output - output, 2))

def dacti_mse(elvart_output, output):
    return 2 * (output - elvart_output)/np.size(elvart_output)


network = [
    Retegek(len(train_x[0]), 10),
    Tanh(),
    Retegek(10, 7),
    Tanh(),
    Retegek(7, len(train_y[0])),
    Tanh()
]

epochs = 100

learning_rate = 0.1

for i in range(epochs):
    error = 0
    for x, y in zip(train_x, train_y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += mse(y, output)
        grad = dacti_mse(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(train_x)
    print("%d/%d, error=%f" % (i + 1, epochs, error))
