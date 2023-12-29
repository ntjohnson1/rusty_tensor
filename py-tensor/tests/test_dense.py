import numpy as np
from rusty_tensor import Dense


def test_constructor():
    d = Dense(np.ones((2, 2)))
    d.__str__()
