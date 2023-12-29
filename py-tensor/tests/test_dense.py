from rusty_tensor import Dense
import pytest
import numpy as np

def test_constructor():
    d = Dense(np.ones((2,2)))
    d.__str__()
