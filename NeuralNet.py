
from GradEngine import Tensor
import random


class Neuron:

    def __init__(self, number_of_inputs):

        self.weights = [Tensor(random.uniform(-1,1)) for x in range(number_of_inputs)]
        self.bias = Tensor(random.uniform(-1,1))


    def __call__(self, x):

       activation = sum (x_index * w_index for x_index, w_index in zip(x, self.weights)) + self.bias
       output = activation._reLU()
       return output


class Layer:

    def __init__(self,number_of_inputs, number_of_neurons):

       self.all_neurons =  [Neuron(number_of_inputs) for x in range(number_of_neurons)]

    def __call__(self,x):

        out = [neuron(x) for neuron in self.all_neurons]
        return out


class FCNN:

    def __init__(self, number_of_inputs, layer_struct):
        total = [number_of_inputs] + layer_struct
        self.layers = [Layer(total[i],total[i+1]) for i in range (len(layer_struct))]

    def __call__(self,x):
        for layer in self.layers:   
            x = layer(x)
        return x

print("=============")

x = [2.0, 3.0, 4.0]
n = FCNN(3,[4,4,1])
print(n(x))
