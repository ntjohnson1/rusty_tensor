use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;
use ndarray::{s, Array, Axis, Ix1, Ix2, Ix3, ShapeBuilder};
use ndarray_linalg::{Norm, Solve};
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
    input_tensor: &'a Dense,
    rank: usize,
    stoptol: f64,
    maxiters: usize,
    dimorder: &'a [usize],
    init: &'a Kruskal,
    printitn: usize,
    fixsigns: bool,
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
        let ndims_max = self.input_tensor.ndims() - 1;
        if !sorted_dims.iter().cloned().eq(0..ndims_max) {
            panic!(
                "Bad dim order. Must contain [0,{:?}] but received {:?}",
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

pub fn cp_als(args: Args) -> Kruskal {
    // Parse args
    let stoptol = args.stoptol;
    let maxiters = args.maxiters;
    let dimorder = args.dimorder;
    let init = args.init;
    let printitn = args.printitn;
    let fixsigns = args.fixsigns;
    let input_tensor = args.input_tensor;
    let rank = args.rank;
    // TODO arg validation

    let ndim = input_tensor.ndims();
    let norm_x = input_tensor.norm();
    let mut model = Kruskal::new();
    let mut factors = init.factor_matrices.clone();
    let mut fit = 0.0;
    // Store the last mttkrp result to accelerate fitness computation
    let mut u_mttkrp =
        Array::<f64, Ix2>::zeros((input_tensor.shape[dimorder[dimorder.len() - 1]], rank).f());

    if printitn > 0 {
        print!("\nCP ALS:\n");
    }

    // Main Loop: Iterate until convergence
    let mut utu = Array::<f64, Ix3>::zeros((rank, rank, ndim).f());
    for (n, factor) in factors.iter().enumerate().take(ndim) {
        utu.slice_mut(s![.., .., n]).assign(&factor.t().dot(factor));
    }

    for iteration in 0..maxiters {
        let fitold = fit;

        let mut weights = Array::<f64, Ix1>::zeros((rank,).f());
        // Iterate over all N modes of the tensor
        for n in dimorder {
            let mut factors_new = input_tensor.mttkrp(&factors, *n);

            // Save the last mttkrp for fitness check.
            if *n == dimorder.len() - 1 {
                u_mttkrp = factors_new.clone();
            }

            let mut y = Array::<f64, Ix2>::ones((rank, rank).f());
            for i in 0..ndim {
                if i != *n {
                    y = y * utu.slice(s![.., .., i]);
                }
            }
            if y.iter().all(|&x| x == 0.0) {
                factors_new = Array::<f64, Ix2>::zeros(factors_new.raw_dim().f());
            } else {
                for i in 0..input_tensor.shape[*n] {
                    // TODO using same y every time so update to more efficient pre-factor
                    let mut update = factors_new.slice_mut(s![i, ..]);
                    update.assign(&y.t().solve(&update.t()).unwrap().t());
                }
            }

            // Normalize each vector to prevent singularities in coeficients
            weights = Array::<f64, Ix1>::zeros((rank,).f());
            if iteration == 0 {
                for i in 0..rank {
                    weights[i] = factors_new.slice(s![i, ..]).norm();
                }
            } else {
                for i in 0..rank {
                    weights[i] = factors_new.slice(s![i, ..]).norm_max().max(1.);
                }
            }
            if !weights.iter().all(|&x| x == 0.0) {
                factors_new = factors_new / weights.clone();
            }

            factors[*n] = factors_new;
            utu.slice_mut(s![.., .., *n])
                .assign(&factors[*n].t().dot(&factors[*n]));
        }
        model = Kruskal::from_data(&weights, &factors);

        // This is equivalent to innerprod(X,P)
        let iprod = ((&model.factor_matrices[dimorder[dimorder.len() - 1]] * &u_mttkrp)
            .sum_axis(Axis(0))
            * weights)
            .sum();

        if norm_x == 0.0 {
            let normresidual = model.norm().powf(2.) - 2. * iprod;
            fit = normresidual;
        } else {
            // The following input can be negative due to rounding
            // and truncation so abs is used
            let normresidual = (norm_x.powf(2.) + model.norm().powf(2.) - 2. * iprod)
                .abs()
                .sqrt();
            fit = 1. - (normresidual / norm_x);
        }
        let fitchange = (fitold - fit).abs();

        // Check for convergence
        let flag = (iteration > 0) && (fitchange < stoptol);

        // TODO some printitn support
        if flag {
            break;
        }
    }
    // Clean up final result

    // TODO our arrange and normalize coupling
    // Arrange the final tensor so columns are normalized.
    //model.arrange(&None);
    model.normalize(&None, &Some(true), &None, &None);

    // Optionally fix signs
    if fixsigns {
        model.fixsigns(&None);
    }

    // TODO printitn support and output dict

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, IxDyn};

    #[test]
    #[should_panic]
    fn empty_interface() {
        // TODO should panic until we can randomly initialize
        let tensor = Dense::new();
        cp_als(ArgBuilder::new(&tensor, 1).build());
    }

    #[test]
    fn smoke_interface() {
        let data: Array<f64, IxDyn> = array![[29.0, 39.0], [63.0, 85.0]].into_dyn();
        let shape: Vec<usize> = vec![2, 2];
        let tensor = Dense::from_data(&data, &Some(shape));
        let weights = array![1.0, 2.0];
        let factors = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[5.0, 6.0], [7.0, 8.0]],
        ];
        let ktensor = Kruskal::from_data(&weights, &factors);
        let model = cp_als(ArgBuilder::new(&tensor, 2).with_init(&ktensor).build());
        print!(
            "Result: {:?} and correct {:?}",
            tensor.data,
            model.full().data
        );
        assert!(tensor.data.abs_diff_eq(&model.full().data, 1e-8));
    }

    #[test]
    fn zero_init() {
        // Confirm we avoid a panic
        let data: Array<f64, IxDyn> = array![[29.0, 39.0], [63.0, 85.0]].into_dyn();
        let shape: Vec<usize> = vec![2, 2];
        let tensor = Dense::from_data(&data, &Some(shape));
        let weights = array![0.0, 0.0];
        let factors = vec![
            array![[0.0, 0.0], [0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
        ];
        let ktensor = Kruskal::from_data(&weights, &factors);
        let model = cp_als(ArgBuilder::new(&tensor, 2).with_init(&ktensor).build());
    }

    #[test]
    fn zero_data() {
        let data: Array<f64, IxDyn> = array![[0.0, 0.0], [0.0, 0.0]].into_dyn();
        let shape: Vec<usize> = vec![2, 2];
        let tensor = Dense::from_data(&data, &Some(shape));
        let weights = array![0.0, 0.0];
        let factors = vec![
            array![[0.0, 0.0], [0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
        ];
        let ktensor = Kruskal::from_data(&weights, &factors);
        let model = cp_als(ArgBuilder::new(&tensor, 2).with_init(&ktensor).build());
    }
}
