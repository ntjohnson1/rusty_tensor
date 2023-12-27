use pyo3::prelude::*;

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadwriteArray2, ToPyArray};
use pyo3::{types::PyIterator, PyAny, PyResult};
use rusty_tensor::utils::khatrirao::khatrirao as _khatrirao;

#[pyfunction]
pub fn khatrirao<'py>(py: Python<'py>, factors: &PyAny) -> PyResult<&'py PyArray2<f64>> {
    let mut extracted = Vec::<Array2<f64>>::new();
    for res in PyIterator::from_object(factors)? {
        if let Ok(mut a) = res?.extract::<PyReadwriteArray2<f64>>() {
            extracted.push(a.as_array_mut().to_owned());
        } else {
            panic!("Defeat");
        }
    }
    Ok(_khatrirao(&extracted, &None).to_pyarray(py))
}
