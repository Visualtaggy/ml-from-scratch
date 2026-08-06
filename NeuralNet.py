
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


print("=============")

x = [2.0, 3.0]
n = Layer(2,4)
print(n(x))
