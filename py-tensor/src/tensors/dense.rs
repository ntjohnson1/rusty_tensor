use pyo3::prelude::*;

use numpy::PyReadwriteArrayDyn;

use rusty_tensor::tensors::dense::Dense as _Dense;

#[pyclass]
pub struct Dense {
    _dense: _Dense,
}

#[pymethods]
impl Dense {
    #[new]
    pub fn new<'py>(mut data: PyReadwriteArrayDyn<'py, f64>) -> Self {
        let data = data.as_array_mut();
        Dense {
            _dense: _Dense::from_data(&data.to_owned(), &None),
        }
    }

    pub fn ndims(&self) -> usize {
        self._dense.ndims()
    }

    pub fn norm(&self) -> f64 {
        self._dense.norm()
    }

    pub fn __str__(&self) -> String {
        format!("{:?}", self._dense)
    }

    // Technically not valid repr but close enough for now
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}
