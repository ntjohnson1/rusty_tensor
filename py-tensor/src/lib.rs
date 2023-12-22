pub mod tensors;

use std::ops::Add;

use crate::tensors::dense::Dense;
use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyIndexError,
    pymodule,
    types::{PyDict, PyModule},
    FromPyObject, PyAny, PyObject, PyResult, Python,
};
use rusty_tensor::tensors::dense::Dense as _Dense;

#[pymodule]
fn tensor_ext<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // Add tensor class
    m.add_class::<Dense>()?;

    Ok(())
}
