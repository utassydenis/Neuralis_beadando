import numpy as np
from sklearn.metrics import mean_squared_error

def activation_tanh(x):
    return np.tanh(x)


def dactivation_tanh(x):
    return 1.0 - x**2


def activation_sigmoid(x):
    return 1/(1+np.exp(-x))


def dactivation_sigmoid(x):
    return x*(1.0-x)

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



# Neurális háló
network = [len(train_x[0]), 7, 8, 10, len(train_y[0])]

sulyok = [np.linspace(-1, 1, network[i]*network[i+1]).reshape((network[i+1], network[i])) for i in range(len(network)-1)]

# Létrehozzuk a súlyokat(és biast) a layerek neuronjai alapján

biases = [np.linspace(-1, 1, (len(network)-1)*1)]
biases = biases[0]


#Training
epoch = 0
learning_rate = 0.001
sumerr = len(train_x)

ml = [np.zeros((network[l+1])) for l in range(len(network)-1)]
vl = [np.zeros((network[l+1])) for l in range(len(network)-1)]
delta = [np.zeros((network[l+1])) for l in range(len(network)-1)]


#while sumerr/len(train_x) >= 0.0000001 and epoch < 200:
while epoch < 500:
    sumerr = 0.0
    epoch += 1
    for input, expected_output in zip(train_x, train_y):
        # forward propagation start
        dactivated_input = []
        activated_input = []
        for suly, bias in zip(sulyok, biases):
            input = (np.dot(input.T, suly.T)+bias).T  # mátrix szorzás + bias
            input = activation_sigmoid(input)  # eredmény aktiválása

            # backpropnak
            dactivated_input.append(dactivation_sigmoid(input))   # eredmény deaktiválása és eltárolása a backpropnak
            activated_input.append(input)

            #backpropnak

        predicted_output = input
        # forward propagation end

        # backprop start
        error = expected_output - predicted_output  # hiba nagysága
        delta_output = error * dactivated_input[-1]


        for i in reversed(range(len(network)-1)):
            if i == len(network) - 2:
                delta[i][:] = error * dactivated_input[-1]
            else:
                np.dot(delta[i + 1], sulyok[i + 1], out=delta[i])
                delta[i] *= dactivated_input[i+1][0]



        for i in reversed(range(len(network)-1)):
            gradiens = np.dot(activated_input[i], delta[i].reshape(len(delta[i]),1).T)
            ml[i] = 0.9*ml[i]+(1-0.9)*np.array(gradiens)
            vl[i] = 0.999 * vl[i] + (1-0.999)*np.power(gradiens,2)
            mh = ml[i]/(1-np.power(0.9, epoch+1))
            vh = vl[i]/(1-np.power(0.999, epoch+1))
            divided = np.divide(np.dot(learning_rate, mh), (np.sqrt(vh) + 0.00000001))
            sulyok[i] = np.add(np.array(sulyok[i]).T,divided[0])
            sulyok[i] = sulyok[i].T


    sumerr += sum([error[j] ** 2 for j in range(network[-1])])
    print(epoch, sumerr)

sumerr = 0.0
y2 = []
for input in test_x:
    for suly, bias in zip(sulyok, biases):
        input = (np.dot(input.T, suly.T) + bias).T  # mátrix szorzás + bias
        input = activation_sigmoid(input)  # eredmény aktiválása
    y2.append(activation_sigmoid(input[0]))

for predicted , expected in zip(y2,test_y):
    print(predicted)
    print(expected)
    print("")
print(mean_squared_error(y2, test_y))
"""
        # biasok beállítása
        for delta, bias in zip(reversed(delta), biases):
            bias += np.sum(delta, axis=0)*learning_rate


    sumerr += sum([error[j] ** 2 for j in range(network[-1])])
    print(epoch, sumerr)



sumerr = 0.0
y2 = []
for input in test_x:
    for suly, bias in zip(sulyok, biases):
        input = (np.dot(input.T, suly.T) + bias).T  # mátrix szorzás + bias
        input = activation_sigmoid(input)  # eredmény aktiválása
    y2.append(activation_sigmoid(input[0]))
print(y2)
print(test_y)
print(mean_squared_error(y2,test_y))
"""