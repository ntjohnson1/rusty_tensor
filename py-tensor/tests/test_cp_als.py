import numpy as np
import pyttb as ttb
from rusty_tensor import Dense, Kruskal, cp_als


def test_constructor():
    data = np.array([[29.0, 39.0], [63.0, 85.0]])
    rank = 2
    d = Dense(np.array([[29.0, 39.0], [63.0, 85.0]]))
    weights = np.array([1.0, 2.0])
    factors = (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]))
    k = Kruskal(weights, factors)
    result = cp_als(d, rank, init=k)
    np.testing.assert_allclose(result.full().data, data)


def test_comparison():
    rank = 2
    shape = (4, 4, 4)
    data = np.random.random(shape)
    weights = np.ones((rank,))
    factors = tuple(np.random.random((first, rank)) for first in shape)
    py_tensor = ttb.tensor(data)
    py_ktensor = ttb.ktensor(list(factors), weights)
    rust_dense = Dense(data)
    rust_kruskal = Kruskal(weights, factors)

    pyttb_result, _, _ = ttb.cp_als(py_tensor, rank, init=py_ktensor, printitn=1)
    rusty_result = cp_als(rust_dense, rank, init=rust_kruskal)
    np.testing.assert_allclose(pyttb_result.full().data, rusty_result.full().data)
