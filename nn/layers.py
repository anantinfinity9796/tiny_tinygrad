# Create the Neuron class
import numpy as np
from engine.tensor import Tensor


class Neuron:
    """ This class create Neuron objects which are the simplest working units in a Neural Network. """

    def __init__(self, nin):
        """ This function initializes the Neuron objecs with the weights and biases according to the number of inputs """
        l = [Tensor(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.w = np.array(l)
        self.b = Tensor(np.random.uniform(-1, 1))

    def __call__(self, x):
        """ This method makes the object act like a function so that anything we define under here will be executed when onject is called. """
        act = np.sum(self.w * np.array(x), initial=self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return np.concatenate((self.w, np.array([self.b])))


# Create the layer class
class Dense:
    """ This class creates Dense or Fully Connected layers where each neuron is connected to each of the inputs.
        1. This does not require any additional dependencies of other classes.
        2. Produces a Dense layer which could be added in the sequential layer to define a Model."""

    def __init__(self, nin, nout):
        """ Create a layer object with Neurons """
        self.neurons = np.array([Neuron(nin) for _ in range(nout)])

    def __call__(self, x):
        """ This function independently evaluates the neurons in the initialized layer. """
        out = np.array([n(x) for n in self.neurons])
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return np.array(params)


class Conv2D:
    pass


class Conv1D:
    pass


class RNN:
    pass


class MaxPooling2D:
    pass


# Create the Sequential Class
class Sequential(Dense):
    def __init__(self, layers=list()):
        """ Initialize a Sequential object with the layer list provided in the parameters"""
        super().__init__()
        self.layers = np.array(layers)

    def add(self, layer):
        return self.layers.append(layer)

    def __call__(self):
        # layers = []
        pass