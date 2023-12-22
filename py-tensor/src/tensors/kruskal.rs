use pyo3::prelude::*;

use numpy::ndarray::Array2;
use numpy::{PyReadwriteArray1, PyReadwriteArray2};
use pyo3::{types::PyIterator, PyAny, PyResult};
use rusty_tensor::tensors::kruskal::Kruskal as _Kruskal;

#[pyclass]
pub struct Kruskal {
    _kruskal: _Kruskal,
}

#[pymethods]
impl Kruskal {
    #[new]
    pub fn new<'py>(mut weights: PyReadwriteArray1<'py, f64>, factors: &PyAny) -> PyResult<Self> {
        let weights = weights.as_array_mut();
        let mut extracted = Vec::<Array2<f64>>::new();
        //let factors: &[Array2<f64>] = factors.extract()?;
        for res in PyIterator::from_object(factors)? {
            if let Ok(mut a) = res?.extract::<PyReadwriteArray2<f64>>() {
                extracted.push(a.as_array_mut().to_owned());
            } else {
                panic!("Defeat");
            }
        }
        Ok(Kruskal {
            _kruskal: _Kruskal::from_data(&weights.to_owned(), &extracted),
        })
    }

    pub fn ndims(&self) -> usize {
        self._kruskal.ndims()
    }

    pub fn norm(&self) -> f64 {
        self._kruskal.norm()
    }

    pub fn __str__(&self) -> String {
        format!("{:?}", self._kruskal)
    }

    // Technically not valid repr but close enough for now
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}
