from tensor_ext import Dense, Kruskal, cp_als
import pytest
import numpy as np

def test_constructor():
    d = Dense(np.array([[29.0, 39.0], [63.0, 85.0]]))
    weights = np.array([1.0, 2.0])
    factors = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]])
    )
    k = Kruskal(weights, factors)
    cp_als(d, k)
