use pyo3::prelude::*;

use crate::tensors::{dense::Dense, kruskal::Kruskal};
use rusty_tensor::cp::cp_als::cp_als as _cp_als;

#[pyfunction]
pub fn cp_als(input_tensor: &Dense, init: &Kruskal, rank: usize) -> Kruskal {
    Kruskal {
        _kruskal: _cp_als(
            &input_tensor._dense,
            rank,
            None,
            None,
            None,
            Some(&init._kruskal),
            None,
            None,
        ),
    }
}
