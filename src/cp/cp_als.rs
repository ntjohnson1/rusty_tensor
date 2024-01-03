use crate::cp::cp_als_args::Args;
use crate::tensors::kruskal::Kruskal;
use ndarray::{s, Array, Axis, Ix1, Ix2, Ix3, ShapeBuilder};
use ndarray_linalg::{Factorize, Norm, Solve};

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
        println!("CP ALS:");
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
                let f = y.t().factorize().unwrap();
                factors_new
                    .axis_iter_mut(Axis(0))
                    .for_each(|mut slice| slice.assign(&f.solve(&slice.t()).unwrap().t()));
            }

            // Normalize each vector to prevent singularities in coeficients
            if iteration == 0 {
                weights = factors_new
                    .axis_iter(Axis(1))
                    .map(|slice| slice.norm())
                    .collect();
            } else {
                weights = factors_new
                    .axis_iter(Axis(1))
                    .map(|slice| slice.norm_max().max(1.))
                    .collect();
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

        if (printitn > 0) && (iteration % printitn == 0 || flag) {
            println!(
                "\tIter {:?}: f = {:1.6e} f-delta = {:7.1e}",
                iteration, fit, fitchange
            );
        }

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

    if printitn > 0 {
        println!("\tFinal f = {:1.6e}", fit);
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::cp_als_args::ArgBuilder;
    use crate::tensors::dense::Dense;
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
