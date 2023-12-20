use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;
use ndarray::{s, Array, Axis, Ix1, Ix2, Ix3};
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

    let N = input_tensor.ndims();
    let normX = input_tensor.norm();
    let mut M = Kruskal::new();
    let mut U = init.factor_matrices.clone();
    let mut fit = 0.0;
    // Store the last mttkrp result to accelerate fitness computation
    let mut U_mttkrp =
        Array::<f64, Ix2>::zeros((input_tensor.shape[dimorder[dimorder.len() - 1]], rank));

    if printitn > 0 {
        print!("CP ALS:");
    }

    // Main Loop: Iterate until convergence
    let mut UtU = Array::<f64, Ix3>::zeros((rank, rank, N));
    for n in 0..N {
        UtU.slice_mut(s![.., .., n]).assign(&U[n].t().dot(&U[n]));
    }

    for iteration in 0..maxiters {
        let fitold = fit;

        let mut weights = Array::<f64, Ix1>::zeros((rank,));
        // Iterate over all N modes of the tensor
        for n in dimorder {
            let mut Unew = input_tensor.mttkrp(&U, *n);

            // Save the last mttkrp for fitness check.
            if *n == dimorder.len() - 1 {
                U_mttkrp = Unew.clone();
            }

            let mut Y = Array::<f64, Ix2>::ones((rank, rank));
            for i in 0..N {
                if i != *n {
                    Y = Y * UtU.slice(s![.., .., i]);
                }
            }

            if Y.abs_diff_eq(&Array::<f64, Ix2>::zeros((rank, rank)), 1e-8) {
                Unew = Array::<f64, Ix2>::zeros(Unew.raw_dim());
            } else {
                for i in 0..N {
                    // TODO using same Y every time so update to more efficient pre-factor
                    let mut update = Unew.slice_mut(s![i, ..]);
                    update.assign(&Y.t().solve(&update.t()).unwrap().t());
                }
            }

            // Normalize each vector to prevent singularities in coeficients
            weights = Array::<f64, Ix1>::zeros((rank,));
            if iteration == 0 {
                for i in 0..rank {
                    weights[i] = Unew.slice(s![i, ..]).norm();
                }
            } else {
                for i in 0..rank {
                    weights[i] = Unew.slice(s![i, ..]).norm_max().max(1.);
                }
            }
            if !weights.abs_diff_eq(&Array::<f64, Ix1>::zeros((rank,)), 1e-8) {
                Unew = Unew / weights.clone();
            }

            U[*n] = Unew;
            //FIXME: Left off on defining this update
            UtU.slice_mut(s![.., .., *n]).assign(&U[*n].t().dot(&U[*n]));
        }
        M = Kruskal::from_data(&weights, &U);

        // This is equivalent to innerprod(X,P)
        let iprod = ((&M.factor_matrices[dimorder[dimorder.len() - 1]] * &U_mttkrp)
            .sum_axis(Axis(0))
            * weights)
            .sum();

        if normX == 0.0 {
            let normresidual = M.norm().powf(2.) - 2. * iprod;
            fit = normresidual.clone();
        } else {
            // The following input can be negative due to rounding
            // and truncation so abs is used
            let normresidual = M.norm().powf(2.) - 2. * iprod;
            fit = normresidual.clone();
        }
        let fitchange = (fitold - fit).abs();

        // Check for convergence
        // TODO make flag just be bool condition
        let flag = (iteration > 0) && (fitchange < stoptol);

        // TODO some printitn support
        if flag {
            break;
        }
    }
    // Clean up final result

    // Arrange the final tensor so columns are normalized.
    M.arrange(&None);

    // Optionally fix signs
    if fixsigns {
        M.fixsigns(&None);
    }

    // TODO printitn support and output dict

    M
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_interface() {
        let tensor = Dense::new();
        cp_als(&tensor, 1, None, None, None, None, None, None);
    }
}
