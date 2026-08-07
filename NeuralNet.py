from GradEngine import Tensor
import random


class Neuron:

    def __init__(self, number_of_inputs, nonlin=True):
        self.weights = [Tensor(random.uniform(-1,1)) for x in range(number_of_inputs)]
        self.bias = Tensor(random.uniform(-1,1))
        self.nonlin = nonlin

    def __call__(self, x):
        activation = sum(x_index * w_index for x_index, w_index in zip(x, self.weights)) + self.bias
        return activation._reLU() if self.nonlin else activation

    def parameters(self):
        return self.weights + [self.bias]


class Layer:

    def __init__(self, number_of_inputs, number_of_neurons, nonlin=True):
        self.all_neurons = [Neuron(number_of_inputs, nonlin) for x in range(number_of_neurons)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.all_neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.all_neurons for p in neuron.parameters()]


class FCNN:

    def __init__(self, number_of_inputs, layer_struct):
        total = [number_of_inputs] + layer_struct
        self.layers = [
            Layer(total[i], total[i+1], nonlin=(i != len(layer_struct)-1))
            for i in range(len(layer_struct))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


print("=============")

xs = [
    [1.5, -2.0, 0.5],
    [-1.0, 1.0, 2.0],
    [2.0, 2.0, -1.5],
    [-0.5, -1.0, -1.0],
]
ys = [1.0, 0.0, 1.0, 0.0]

network = FCNN(3, [4,4,1])
y_hat_final = [network(x) for x in xs]
for pred, target in zip(y_hat_final, ys):
    print(f"pred={pred.data:.4f}  target={target}")
    
for epoch in range(20):
    y_hat = [network(x) for x in xs]
    loss = sum((pred - ygt)**2 for ygt, pred in zip(ys, y_hat))
    for p in network.parameters():
        p.grad = 0.0
    loss.backward()
    for p in network.parameters():
        p.data += -0.01 * p.grad
    print(epoch, loss.data)

y_hat_final = [network(x) for x in xs]
for pred, target in zip(y_hat_final, ys):
    print(f"pred={pred.data:.4f}  target={target}")