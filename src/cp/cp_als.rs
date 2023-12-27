use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;
use ndarray::{s, Array, Axis, Ix1, Ix2, Ix3, ShapeBuilder};
use ndarray_linalg::{Norm, Solve};

//TODO add builder pattern here for argument simplicity/explicitness

pub fn cp_als(
    input_tensor: &Dense,
    rank: usize,
    stoptol: Option<f64>,
    maxiters: Option<usize>,
    dimorder: Option<&[usize]>,
    init: Option<&Kruskal>,
    printitn: Option<usize>,
    fixsigns: Option<bool>,
) -> Kruskal {
    // Parse args
    let stoptol = stoptol.unwrap_or(1e-4);
    let maxiters = maxiters.unwrap_or(1000);
    let default_dimorder = (0..input_tensor.ndims()).collect::<Vec<usize>>();
    let dimorder = dimorder.unwrap_or(&default_dimorder);
    // TODO: init properly
    let init = init.unwrap();
    let printitn = printitn.unwrap_or(1);
    let fixsigns = fixsigns.unwrap_or(false);
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
        cp_als(&tensor, 1, None, None, None, None, None, None);
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
        let model = cp_als(&tensor, 2, None, None, None, Some(&ktensor), None, None);
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
        let model = cp_als(&tensor, 2, None, None, None, Some(&ktensor), None, None);
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
        let model = cp_als(&tensor, 2, None, None, None, Some(&ktensor), None, None);
    }
}
