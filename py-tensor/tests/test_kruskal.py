from tensor_ext import Kruskal
import pytest
import numpy as np

def test_constructor():
    weights = np.array([1.0, 1.0])
    factors = (np.ones((2,2)), 2.0 * np.ones((2,2)))
    k = Kruskal(weights, factors)
    k.__str__()
