#Code for Testing of the Tensor Class
from engine.tensor import Tensor
import numpy as np
import pytest
from nn import layers

# Because we want to test our Tensor class we will have to initialize tensors again and again.
# So we would just create a fixture that would create Tensors for whichever test needs it.
# It's standard practice to define fixtures at the start of the file.


@pytest.fixture
def create_tensors():
    """ This fixture for creating a fixed number of tensors which are passed in the argument above n_tensors.
        By default it creates one tensor of the shape(1,4)"""
    tensor_list = []
    shape = (1, 4)
    for i in range(2):
        label = f"Tensor_{np.random.randint(low=0, high=10)}"
        data = np.random.randint(low=0, high=500, size=shape)
        tensor_list.append(Tensor(data=data, label=label))
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
    ((2.1, 3.4, 5.9), 'Tensor_3')

])
def test_initialization(data, label):
    tensor = Tensor(data=data, label=label)
    print(tensor)
    assert isinstance(tensor, Tensor)
    assert isinstance(tensor.data, np.ndarray)
    assert isinstance(tensor.label, str)

# Testing the addition of two tensor objects
def test_addtensor(create_tensors):
    tensor_list = create_tensors
    assert isinstance(tensor_list[0], Tensor)
    assert(tensor_list[0].shape == (1, 4))

