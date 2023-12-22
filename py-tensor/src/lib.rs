pub mod tensors;

use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;

use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn tensor_ext<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // Add tensor classes
    m.add_class::<Dense>()?;
    m.add_class::<Kruskal>()?;

    Ok(())
}
