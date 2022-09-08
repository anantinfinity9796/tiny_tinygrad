""" This will be a framework built from scratch using python and numpy. """
import numpy as np


# Create the Tensor class for the Framework
class Tensor(object):
    """ This creates the Tensor object with the data. """

    def __init__(self, data, _children=(), _op='', label=''):
        """ Creates and Initialize the Tensor object with the values provided. """

        self.data = np.array(data)  # Initialize the data
        self.grad = 0.0  # Initialize the gradients to  0.0
        self._backward = lambda: None  # This initializes an empty backward function for each Tensor. By default it does not do anything
        self._prev = set(
            _children)  # Initialize the children of the network to be used in backprop.They should be unique hence set is used.
        self._op = _op  # Initialize the operation that created the tensor
        self.label = label  # Initialize the label of the tensor
        self.shape = self.data.shape  # Initialize the shape of the Tensor

    def __repr__(self) -> str:
        """ Used to display the object in python. This is the representation of the object. """

        return f"Tensor(data = {self.data}, label={self.label})"

    def __str__(self):
        """ This handles how the object would look like when it is printed """

        return f"Tensor(data = {self.data})"

    def __add__(self, other):
        """ Used to add two tensors """

        # We need to perform a check here if the other value is a Tensor or NOT, if NOT then we need to convert it to a tensor to perform the operation
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        # So the out values are an addition of self and other. So we want to set out._backward to be the function that backpropagates the gradient.
        # So now lets define what happens when we call out.grad
        # Our job is to take out.grad and backpropagate into self.grad and other.grad because out = self+other.
        # So we backpropagate it by the chain rule i.e  local derivative * global gradient
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward  # Here we are assigning the _backward function defined in the addition to the output._backward of the output tensor.
        # By doing the assign like this we would be able to backpropagate easily.
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """ Function to subtract two tensors"""

        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-1) * other

    def __rsub__(self, other):
        """ Used for reverse substraction if the first operand is not a Tensor then the first is subtracted from the second."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-1) * other

    def __mul__(self, other):
        """ Used to multiply two tensors. """
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        """ Used for reverse multilication, same as __rsub__ and __radd__ methods. """
        return self * other

    def __neg__(self):
        """ Used to get the negative of a tensor i.e multiply the tensor by -1. """

        out = (-1) * self
        return out

    def __pow__(self, pow):
        """ This raise the data of the tensor to a power. """

        assert isinstance(pow, (int, float))
        out = Tensor(self.data ** pow, (self,), _op=f'**{pow}')

        def _backward():
            """ The function which calculates the backpropagated gradient of the pow calculation. """
            self.grad += pow * (self.data ** (pow - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """ This calculates the exponent of the tensor i.e (e**self) """

        x = self.data  # take out the data of the Tensor
        out = Tensor(np.exp(x), (self,), _op='exp')

        def _backward():
            """ The function which calculates the backpropagated gradient of the exponentiation operation. """
            self.grad += np.exp(x) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """ This calculate the divison of two tensors """
        return self * (other ** (-1.0))

    # Now we are going to write the most important function of the Tensor class i.e the backwards function
    # This function sorts the tensor and its children topologically and backpropagates through them to their inputs.
    # This helps us in a way that it organizes the nodes of the function in the topological order so we can backpropagate through the nodes.
    # When backproping through this way we get to calculate the gradients of the nodes which the current node is dependent on.

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones(())

        for node in reversed(topo):
            node._backward()

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = Tensor(t, (self,), label='tanh')

        def _backward():
            self.grad += (1 - (t * t)) * out.grad

        out._backward = _backward
        return out

    # Create the Neuron class


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
        # comment
        pass