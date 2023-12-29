[![Coverage Status](https://coveralls.io/repos/github/ntjohnson1/rusty_tensor/badge.svg?branch=main)](https://coveralls.io/github/ntjohnson1/rusty_tensor?branch=main)

# rusty_tensor
Learning About Using Rust From Python via Tensor Toolbox

## Thoughts
1. Python Bindings
    * Python wrappings reside in ./py-tensor
        * Local build via `maturin develop -r` in that directory
    * See [pyO3](https://github.com/PyO3/pyo3) for more details.

1. Pyttb is based on MATLAB.
This is mostly a port from pyttb discarding the MATLAB.
There isn't an attempt to maintain 1:1 compatibility.

1. We depend on ndarray and ndarray-linalg
    * See details on [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg/blob/master/README.md) to get that to build
    * My path on ubuntu was:
       * Install `gfortran`
       * `cargo test --features=openblas`
          * This installs the openblas crate which takes a bit the first time, but minimizes additional setup
