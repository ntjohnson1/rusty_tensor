import pyttb as ttb
from rusty_tensor import Dense, Kruskal, cp_als
from typing import Tuple
import time
import numpy as np

def run_pyttb(source: ttb.tensor, init: ttb.ktensor, rank: int) -> ttb.ktensor:
    M, _, _ = ttb.cp_als(source, rank, init=init, printitn=-1)
    return M

def run_rusty(source:Dense, init: Kruskal, rank:int) -> Kruskal:
    M = cp_als(source, init, rank)
    return M

def benchmark(shape: Tuple[int,...], num_iters: int):
    pyttb_time = 0.0
    rusty_time = 0.0
    for _ in range(num_iters):
        data = np.random.random(shape)
        rank = 2
        weights = np.ones((rank,))
        factors = tuple(np.random.random((first, rank)) for first in shape)
        py_tensor = ttb.tensor(data)
        py_ktensor = ttb.ktensor(list(factors), weights)
        rust_dense = Dense(data)
        rust_kruskal = Kruskal(weights, factors)

        # TODO should provide init here for fair comparison
        start = time.time()
        pyttb_result = run_pyttb(py_tensor, py_ktensor, rank)
        pyttb_time += time.time() - start

        start = time.time()
        rusty_result = run_rusty(rust_dense, rust_kruskal, rank)
        rusty_time += time.time() - start

        np.testing.assert_allclose(pyttb_result.full().data, rusty_result.full().data)
    print(
        f"Pyttb time: {pyttb_time}\n"
        f"Rust time: {rusty_time}\n"
        f"Shape: {shape}"
    )

if __name__=="__main__":
    benchmark((100, 100, 100), 10)
