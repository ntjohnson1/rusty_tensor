use pyo3::exceptions;
use pyo3::prelude::*;

use crate::tensors::{dense::Dense, kruskal::Kruskal};
use numpy::PyReadwriteArray1;
use rusty_tensor::cp::cp_als::{cp_als as _cp_als, ArgBuilder, InitStrategy};

pub enum InitType<'a> {
    // FIXME resolve unpacking to avoid copy
    Ktensor(Kruskal),
    Strategy(&'a str),
}

impl<'source> FromPyObject<'source> for InitType<'_> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract() {
            return Ok(InitType::Ktensor(s));
        } else {
            Err(exceptions::PyTypeError::new_err("Couldn't convert"))
        }
    }
}

#[pyfunction]
pub fn cp_als<'a>(
    input_tensor: &Dense,
    rank: usize,
    stoptol: Option<f64>,
    maxiters: Option<usize>,
    dimorder: Option<PyReadwriteArray1<'a, usize>>,
    init: Option<InitType>,
    printitn: Option<usize>,
    fixsigns: Option<bool>,
) -> Kruskal {
    let mut builder = ArgBuilder::new(&input_tensor._dense, rank);
    if stoptol.is_some() {
        builder.with_stoptol(stoptol.unwrap());
    }
    if maxiters.is_some() {
        builder.with_maxiters(maxiters.unwrap());
    }
    let rust_dimorder: Vec<usize>;
    if dimorder.is_some() {
        rust_dimorder = dimorder.unwrap().as_array().to_vec();
        builder.with_dimorder(&rust_dimorder);
    }
    if printitn.is_some() {
        builder.with_printitn(printitn.unwrap());
    }
    if fixsigns.is_some() {
        builder.with_fixsigns(fixsigns.unwrap());
    }
    let rust_ktensor: Kruskal;
    if init.is_some() {
        match init.unwrap() {
            InitType::Ktensor(ktensor) => {
                rust_ktensor = ktensor;
                builder.with_init(&rust_ktensor._kruskal);
            }
            InitType::Strategy(strategy) => {
                //FIXME update string parsing etc
                builder.with_init_strategy(InitStrategy::Random);
            }
        }
    }
    Kruskal {
        _kruskal: _cp_als(builder.build()),
    }
}
