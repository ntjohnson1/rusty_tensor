pub mod cp;
pub mod tensors;

use crate::cp::cp_als::cp_als;
use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;
use pyo3::prelude::*;

use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn tensor_ext<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // Add tensor classes
    m.add_class::<Dense>()?;
    m.add_class::<Kruskal>()?;
    m.add_function(wrap_pyfunction!(cp_als, m)?)?;

    Ok(())
}
