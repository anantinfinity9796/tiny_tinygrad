#Code for Testing of the Tensor Class
from engine.tensor import Tensor
import numpy as np
import pytest
from nn import layers

# Because we want to test our Tensor class we will have to initialize tensors again and again.
# So we would just create a fixture that would create Tensors for whichever test needs it.
# It's standard practice to define fixtures at the start of the file.


def create_tensors(n_tensors=1, label='', shape=(1,4)) -> list():
    """ This fixture for creating a fixed number of tensors which are passed in the argument above n_tensors.
        By default, it creates one tensor of the shape(1,4)"""
    tensor_list = []
    for i in range(n_tensors):
        if label == '':
            label = f"Tensor_{np.random.randint(low=0, high=10)}"
        data = np.random.randint(low=0, high=500, size=shape)
        tensor_list.append(Tensor(data=data, label=label))
    if len(tensor_list) == 1:
        return tensor_list[0]
    else:
        return tensor_list

    # def _tensor_factory(**kwargs) -> Tensor:
    #     tensor_label = kwargs.pop('label',str(np.random.randint(low=1, high=10000000)))
    #     tensor_data = kwargs.pop('data')
    #     if tensor_data is None:
    #         raise ValueError("No data provided in the Tensor. Object cannot be created")
    #         return
    #     return Tensor(data=tensor_data, label=tensor_label)
    # return _tensor_factory


# Testing if the tensors are getting initialized properly or not
@pytest.mark.parametrize("data, label",
[
    ([1, 2, 3, 4, 5], 'Tensor_1'),
    ([2], 'Tensor_2'),
    ([2, 3.4, 5.9], 'Tensor_3'),
    ((2.1, 3.4, 5.9), 'Tensor_3'),
    ([[1, 2, 3], [2, 3, 4]], 'Tensor_4'),
    ([[[1, 2, 3, 4],
       [2, 3, 4, 5]],
      [[3, 4, 5, 6],
       [1, 2, 4, 5]]
      ], 'Tensor_6')

])
def test_initialization(data, label):
    """ Tests the initialization of the tensors"""
    tensor = Tensor(data=data, label=label)
    assert isinstance(tensor, Tensor)
    assert isinstance(tensor.data, np.ndarray)
    assert isinstance(tensor.label, str)
    assert tensor.shape == tensor.data.shape

# Testing the addition of two tensor objects
def test_add_tensor():
    """ Function for testing the addition of two tensors. """
    # Tests for 2D tensors
    tensor_list = create_tensors(n_tensors=2, shape=(2, 3))
    # creating variables of addition of the tensors
    tensor_add = (tensor_list[0] + tensor_list[1]).data
    ndarray_add = tensor_list[0].data + tensor_list[1].data
    # comparing the arrays
    assert isinstance(tensor_list[0], Tensor)
    assert (tensor_list[0].shape == (2, 3))
    assert np.array_equal(tensor_add, ndarray_add)

    # assert statements for 3D tensors
    mul_tensor_list = create_tensors(n_tensors=2, shape=(2, 2, 3))
    tensor_add = (mul_tensor_list[0] + mul_tensor_list[1]).data
    ndarray_add = mul_tensor_list[0].data + mul_tensor_list[1].data
    # comparing the arrays
    assert isinstance(mul_tensor_list[0], Tensor)
    assert (mul_tensor_list[0].shape == (2, 2, 3))
    assert np.array_equal(tensor_add, ndarray_add)

# Testing the subtraction of two tensors
def test_sub_tensor():
    """ Function for testing the subtraction of two tensors. Because of the way sub is implemented negation is also tested here. """
    # Testing for 2D tensors
    tensor_list = create_tensors(n_tensors=2, shape=(2,3))
    # creating variables of addition of the tensors
    tensor_sub = (tensor_list[0] - tensor_list[1]).data
    ndarray_sub = tensor_list[0].data - tensor_list[1].data
    # comparing the arrays
    assert isinstance(tensor_list[0], Tensor)
    assert (tensor_list[0].shape == (2, 3))
    assert np.array_equal(tensor_sub, ndarray_sub)

    # Testing for 3D tensors
    mul_tensor_list = create_tensors(n_tensors=2, shape=(2, 2, 3))
    tensor_sub = (mul_tensor_list[0] - mul_tensor_list[1]).data
    ndarray_sub = mul_tensor_list[0].data - mul_tensor_list[1].data
    # comparing the arrays
    assert isinstance(mul_tensor_list[0], Tensor)
    assert (mul_tensor_list[0].shape == (2, 2, 3))
    assert np.array_equal(tensor_sub, ndarray_sub)

# Testing the multiplication of two tensors
def test_matmul_tensor():
    """ Function for testing of the matrix-multiplication of two tensors. """
    tensor_1 = create_tensors(n_tensors=1, shape=(2, 3))
    tensor_2 = create_tensors(n_tensors=1, shape=(3, 2))

    # creating variables for the multiplication of the tensors
    tensor_mul = tensor_1 @ tensor_2
    ndarray_mul = tensor_1.data @ tensor_2.data
    reverse = tensor_2 @ tensor_1

    # comparing the tensors
    assert np.array_equal(tensor_mul.data, ndarray_mul)
    assert tensor_mul.shape == ndarray_mul.shape
    assert reverse.shape == (3, 3)
    assert ndarray_mul.shape == (2, 2)


    # Testing for 3D and 4D tensors
    tensor_1 = create_tensors(n_tensors=1, shape=(4,3, 2, 3))
    tensor_2 = create_tensors(n_tensors=1, shape=(4,3, 3, 2))

    tensor_mul = tensor_1 @ tensor_2
    ndarray_mul = tensor_1.data @ tensor_2.data
    reverse_mul = tensor_2 @ tensor_1
    reverse_nd = tensor_2.data @ tensor_1.data
    assert isinstance(reverse_mul, Tensor)
    assert np.array_equal(tensor_mul.data, ndarray_mul)
    assert np.array_equal(reverse_mul.data, reverse_nd)


# Testing the division of two tensors
def test_div_tensor():
    """ Testing the division of two tensors. """
    tensor_1 = create_tensors(n_tensors=1, shape=(2, 2))
    tensor_2 = create_tensors(n_tensors=1, shape=(2, 2))

    tensor_div = tensor_1 / tensor_2
    ndarray_div = tensor_1.data / tensor_2.data
    assert isinstance(tensor_div, Tensor)
    tensor_div = tensor_div.data
    assert isinstance(tensor_div, np.ndarray)
    assert isinstance(ndarray_div, np.ndarray)
    assert tensor_div.shape == (2, 2)
    assert ndarray_div.shape == (2, 2)
    # assert np.array_equal(tensor_div, ndarray_div)

# Testing for a tensor raised to a power
def test_pow():
    """ Testing the tensor elements raised to a power. """
    # Testing for a 2D Tensor
    tensor_1 = create_tensors(n_tensors=1, shape=(4, 2))
    tensor_pow = tensor_1 ** 2
    ndarray_pow = np.power(tensor_1.data, 2)

    assert isinstance(tensor_pow, Tensor)
    assert tensor_pow.shape == (4, 2)
    assert np.array_equal(tensor_pow.data, ndarray_pow)

    # Testing for a 3D tensor
    tensor_1 = create_tensors(n_tensors=1, shape=(3, 4, 2))
    tensor_pow = tensor_1 ** 3
    ndarray_pow = np.power(tensor_1.data, 3)

    assert isinstance(tensor_pow, Tensor)
    assert tensor_pow.shape == (3, 4, 2)
    assert np.array_equal(tensor_pow.data, ndarray_pow)


# Testing the exp function
def test_exp():
    """ Testing the euler's number raised to the tensor i.e. exp(tensor). """
    tensor_1 = create_tensors(n_tensors=1, shape=(4, 2))

    tensor_exp = tensor_1.exp()
    ndarray_exp = np.exp(tensor_1.data)

    assert isinstance(tensor_exp, Tensor)
    assert tensor_exp.shape == (4, 2)
    assert np.array_equal(tensor_exp.data, ndarray_exp)

    # Testing for 3D Tensors
    tensor_1 = create_tensors(n_tensors=1, shape=(4, 2, 1))

    tensor_exp = tensor_1.exp()
    ndarray_exp = np.exp(tensor_1.data)

    assert isinstance(tensor_exp, Tensor)
    assert tensor_exp.shape == (4, 2, 1)
    assert np.array_equal(tensor_exp.data, ndarray_exp)