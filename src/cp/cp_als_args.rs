use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;
use ndarray::{Array, Ix2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug, PartialEq)]
pub enum InitStrategy {
    Random,
    // Add implementation when dense supports nvecs method
    NVecs,
}

#[derive(Debug)]
pub struct Args<'a> {
    pub input_tensor: &'a Dense,
    pub rank: usize,
    pub stoptol: f64,
    pub maxiters: usize,
    pub dimorder: &'a [usize],
    pub init: &'a Kruskal,
    pub printitn: usize,
    pub fixsigns: bool,
}

#[derive(Debug)]
pub struct ArgBuilder<'a> {
    input_tensor: &'a Dense,
    rank: usize,
    stoptol: f64,
    maxiters: usize,
    dimorder: Option<&'a [usize]>,
    init: Option<&'a Kruskal>,
    printitn: usize,
    fixsigns: bool,
    local_dimorder: Vec<usize>,
    init_strategy: InitStrategy,
    local_init: Kruskal,
}

impl<'a> ArgBuilder<'a> {
    pub fn new(input_tensor: &Dense, rank: usize) -> ArgBuilder {
        ArgBuilder {
            input_tensor,
            rank,
            stoptol: 1e-4,
            maxiters: 1000,
            dimorder: None,
            init: None,
            printitn: 1,
            fixsigns: false,
            local_dimorder: Vec::<usize>::new(),
            init_strategy: InitStrategy::Random,
            local_init: Kruskal::new(),
        }
    }

    pub fn with_stoptol(&mut self, stoptol: f64) -> &mut Self {
        self.stoptol = stoptol;
        self
    }

    pub fn with_maxiters(&mut self, maxiters: usize) -> &mut Self {
        self.maxiters = maxiters;
        self
    }

    pub fn with_dimorder(&mut self, dimorder: &'a [usize]) -> &mut Self {
        let mut sorted_dims = dimorder.to_vec();
        sorted_dims.sort_unstable();
        let ndims_max = self.input_tensor.ndims();
        if !sorted_dims.iter().cloned().eq(0..ndims_max) {
            panic!(
                "Bad dim order. Must contain [0,{:?}) but received {:?}",
                ndims_max, dimorder
            );
        }
        self.dimorder = Some(dimorder);
        self
    }

    pub fn with_init(&mut self, init: &'a Kruskal) -> &mut Self {
        if init.ndims() != self.input_tensor.ndims() {
            panic!(
                "Initial guess doesn't have {:?} modes",
                self.input_tensor.ndims()
            );
        }
        if init.ncomponents() != self.rank {
            panic!("Initial guess doesn't have {:?} components", self.rank);
        }
        // TODO dim order check of facto matrices shape
        // can we enforce dimorder set first?
        self.init = Some(init);
        self
    }

    pub fn with_init_strategy(&mut self, init: InitStrategy) -> &mut Self {
        self.init_strategy = init;
        self.init = None;
        self
    }

    pub fn with_printitn(&mut self, printitn: usize) -> &mut Self {
        self.printitn = printitn;
        self
    }

    pub fn with_fixsigns(&mut self, fixsigns: bool) -> &mut Self {
        self.fixsigns = fixsigns;
        self
    }

    fn gen_dimorder(&mut self) {
        if self.dimorder.is_none() {
            self.local_dimorder = (0..self.input_tensor.ndims()).collect::<Vec<usize>>();
        }
    }

    fn gen_init(&mut self) {
        if self.init.is_none() {
            if self.init_strategy == InitStrategy::Random {
                let ndims = self.input_tensor.ndims();
                let mut factors = Vec::<Array<f64, Ix2>>::new();
                for n in 0..ndims {
                    factors.push(Array::random(
                        (self.input_tensor.shape[n], self.rank),
                        Uniform::new(0., 1.),
                    ));
                }
                self.local_init = Kruskal::from_factor_matrices(&factors);
            } else {
                panic!("Unsupported init strategy");
            }
        }
    }

    fn get_dimorder(&self) -> &[usize] {
        match self.dimorder {
            None => &self.local_dimorder,
            Some(reference) => reference,
        }
    }

    fn get_init(&self) -> &Kruskal {
        match self.init {
            None => &self.local_init,
            Some(reference) => reference,
        }
    }

    pub fn build(&mut self) -> Args {
        self.gen_dimorder();
        self.gen_init();
        Args {
            input_tensor: self.input_tensor,
            rank: self.rank,
            stoptol: self.stoptol,
            maxiters: self.maxiters,
            dimorder: self.get_dimorder(),
            init: self.get_init(),
            printitn: self.printitn,
            fixsigns: self.fixsigns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, IxDyn};

    fn get_dense() -> Dense {
        let data: Array<f64, IxDyn> = array![[29.0, 39.0], [63.0, 85.0]].into_dyn();
        let shape: Vec<usize> = vec![2, 2];
        Dense::from_data(&data, &Some(shape))
    }

    #[test]
    fn check_defaults() {
        let tensor = get_dense();
        let mut builder = ArgBuilder::new(&tensor, 2);
        let args = builder.build();
        assert!(args.stoptol == 1e-4);
        assert!(args.maxiters == 1000);
        assert!(args.printitn == 1);
        assert!(args.fixsigns == false);
    }

    #[test]
    fn set_primitive_values() {
        let tensor = get_dense();
        let expected_stoptol = 0.123;
        let expected_maxiters = 1234;
        let expected_printitn = 1234;
        let expected_fixsigns = true;
        let mut builder = ArgBuilder::new(&tensor, 2);
        builder
            .with_stoptol(expected_stoptol)
            .with_maxiters(expected_maxiters)
            .with_printitn(expected_printitn)
            .with_fixsigns(expected_fixsigns);
        let args = builder.build();
        assert!(args.stoptol == expected_stoptol);
        assert!(args.maxiters == expected_maxiters);
        assert!(args.printitn == expected_printitn);
        assert!(args.fixsigns == expected_fixsigns);
    }

    #[test]
    fn set_dimorder() {
        let tensor = get_dense();
        let mut builder = ArgBuilder::new(&tensor, 2);
        let dimorder = (0..tensor.ndims()).collect::<Vec<usize>>();
        builder.with_dimorder(&dimorder);
        let args = builder.build();
        assert!(args.dimorder == dimorder);
    }

    #[should_panic]
    #[test]
    fn bad_dimorder() {
        let tensor = get_dense();
        let mut builder = ArgBuilder::new(&tensor, 2);
        // Dim order missing dims
        let dimorder = (0..tensor.ndims() - 1).collect::<Vec<usize>>();
        builder.with_dimorder(&dimorder);
    }

    #[test]
    fn random_init() {
        let tensor = get_dense();
        let mut builder = ArgBuilder::new(&tensor, 2);
        let init_strategy = InitStrategy::Random;
        builder.with_init_strategy(init_strategy);
        let args = builder.build();
    }

    #[should_panic]
    #[test]
    fn nvecs_unsupported() {
        let tensor = get_dense();
        let mut builder = ArgBuilder::new(&tensor, 2);
        let init_strategy = InitStrategy::NVecs;
        builder.with_init_strategy(init_strategy);
        builder.build();
    }

    #[should_panic]
    #[test]
    fn bad_kruskal_init_ndims() {
        let tensor = get_dense();
        let weights = array![0.0, 0.0];
        let factors = vec![array![[0.0, 0.0], [0.0, 0.0]]];
        let ktensor = Kruskal::from_data(&weights, &factors);
        let mut builder = ArgBuilder::new(&tensor, 2);
        builder.with_init(&ktensor);
        builder.build();
    }

    #[should_panic]
    #[test]
    fn bad_kruskal_init_ncomponents() {
        // TODO this can probably be more rigidly controlled by ktensor struct instead of in algorithm
        let tensor = get_dense();
        let weights = array![0.0, 0.0];
        let factors = vec![
            array![[0.0, 0.0], [0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
        ];
        let mut ktensor = Kruskal::from_data(&weights, &factors);
        ktensor.weights = array![0.0, 0.0, 0.0];
        let mut builder = ArgBuilder::new(&tensor, 2);
        builder.with_init(&ktensor);
        builder.build();
    }
}
