""" This will be a framework built from scratch using python and numpy. """
import numpy as np

# TODO : Add support for multi-dimesional Tensors. Currently this class is for scalar Tensors. Add them for multiple dimensions.

# Create the Tensor class for the Framework
class Tensor(object):
    """ This creates the Tensor object with the data. """

    def __init__(self, data, _children=(), _op='', label=None):
        """ Creates and Initialize the Tensor object with the values provided. """

        tensor_data = np.array(data)
        self.data = tensor_data  # Initialize the data
        self.shape = tensor_data.shape
        self.grad = np.zeros(tensor_data.shape)  # Initialize the gradients to  0.0
        self._backward = lambda: None  # This initializes an empty backward function for each Tensor. By default, it does not do anything
        self._prev = set(
            _children)  # Initialize the children of the network to be used in backprop.They should be unique hence set is used.
        self._op = _op  # Initialize the operation that created the tensor

        if label == None:
            d_label = "Tensor_"+str(np.random.randint(1, 10000000))  # Create a dummy label if label not provided
            self.label = d_label
        else:
            self.label = label  # Initialize the label of the tensor




    def __repr__(self) -> str:
        """ Used to display the object in python. This is the representation of the object. """

        return f"Tensor(data = {self.data}, label={self.label}, shape={self.data.shape})"

    def __str__(self):
        """ This handles how the object would look like when it is printed """

        return f"Tensor(data = {self.data})"

    def __add__(self, other):
        """ Used to add two tensors """

        # We need to perform a check here if the other value is a Tensor or NOT, if NOT then we need to convert it to a tensor to perform the operation
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        # So the out values are an addition of self and other. So we want to set out._backward to be the function that back-propagates the gradient.
        # So now let's define what happens when we call out.grad
        # Our job is to take out.grad and backpropagate into self.grad and other.grad because out = self+other.
        # So we backpropagate it by the chain rule i.e.  local derivative * global gradient
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
        """ Used for reverse subtraction if the first operand is not a Tensor then the first is subtracted from the second."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-1) * other

    def __mul__(self, other):
        """ Used to do elementwise multiplication of two tensors. It uses the * operator symbol.
         This represents the elementwise-multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = np.multiply(self.data, other.data)
        out = Tensor(data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """ Function for the matrix multiplication of two tensors. It uses the @ operator.
        This represents the matrix-multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = np.matmul(self.data, other.data)
        out = Tensor(data, (self, other), '@')

        def _backward():
            self.grad += np.matmul(other.data,  out.grad)
            other.grad += np.matmul(self.data, out.grad)

        out._backward = _backward
        return out
    def __rmatmul__(self, other):
        """ Used for the reverse matrix multiplication."""
        return self @ other

    def __rmul__(self, other):
        """ Used for reverse multiplication, same as __rsub__ and __radd__ methods. """
        return self * other

    def __neg__(self):
        """ Used to get the negative of a tensor i.e. multiply the tensor by -1. """

        out = (-1) * self
        return out

    def __pow__(self, pow):
        """ This raise the data of the tensor to a power. The " pow " argument here is the power that the tensor is raised to. """

        assert isinstance(pow, (int, float))
        out = Tensor(self.data ** pow, (self,), _op=f'**{pow}')

        def _backward():
            """ The function which calculates the back-propagated gradient of the pow calculation. """
            self.grad += pow * (self.data ** (pow - 1)) @ out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """ This calculates the divison of two tensors """
        if np.array_equal(other.data, np.zeros((other.shape))) or other == 0:
            raise ZeroDivisionError("Dividing by zero is not permitted")
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** (-1.0))

    def exp(self):
        """ This calculates the exponent of the tensor i.e (e**self) """

        x = self.data  # take out the data of the Tensor
        out = Tensor(np.exp(x), (self,), _op='exp')

        def _backward():
            """ The function which calculates the back-propagated gradient of the exponentiation operation. """
            self.grad += np.exp(x) @ out.grad

        out._backward = _backward
        return out

# TODO: Find a way to implement the backwards pass of the dot product for tensors of different shapes.
    # Error message : operands could not be broadcast together with shapes (2,3) (2,2)
    def dot(self, other):
        """ This calculates the dot product of two tensors. """
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(data=np.dot(self.data, other.data), _children=(self, other), _op='dot')

        def _backward():
            # self.grad += other.data * out.grad
            other.grad += self.data @ out.grad

        out._backward = _backward()
        return out
    # Now we are going to write the most important function of the Tensor class i.e. the backwards function
    # This function sorts the tensor and its children topologically and back-propagates through them to their inputs.
    # This helps us in a way that it organizes the nodes of the function in the topological order, so we can backpropagate through the nodes.
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

        self.grad = np.ones(self.data.shape)

        for node in reversed(topo):
            node._backward()

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = Tensor(t, (self,), label='tanh')

        def _backward():
            self.grad += (1 - (t @ t)) @ out.grad

        out._backward = _backward
        return out



