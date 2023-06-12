use crate::utils::ndarray_helpers::{max_abs, p_norm, sign};
use ndarray::{s, Array, Array2, Ix1, Ix2};

#[derive(Debug)]
pub struct Kruskal {
    // For now assume TODO throughout to make f64 generic
    weights: Array<f64, Ix1>,
    factor_matrices: Vec<Array<f64, Ix2>>,
}

impl Kruskal {
    pub fn new() -> Self {
        Self {
            weights: Array::<f64, Ix1>::zeros((0,)),
            factor_matrices: Vec::new(),
        }
    }

    pub fn from_data(weights: &Array<f64, Ix1>, factors: &[Array<f64, Ix2>]) -> Self {
        for factor in factors {
            if weights.len() != factor.shape()[1] {
                // TODO print details about mismatch
                panic!("Size of factor matrix does not match number of weights");
            }
        }
        Self {
            weights: weights.clone(),
            factor_matrices: factors.to_vec().clone(),
        }
    }

    pub fn from_factor_matrices(factors: &[Array<f64, Ix2>]) -> Self {
        let ncomponents = factors[0].shape()[1];
        Kruskal::from_data(&Array::<f64, Ix1>::ones((ncomponents,)), &factors)
    }

    pub fn ndims(&self) -> usize {
        self.factor_matrices.len()
    }

    pub fn ncomponents(&self) -> usize {
        self.weights.len()
    }

    pub fn norm(&self) -> f64 {
        // TODO: CoWArrays for row and column weights is probably more readable
        // without copy overhead
        let mut coef_matrix = self
            .weights
            .to_shape((self.ncomponents(), 1))
            .unwrap()
            .dot(&self.weights.to_shape((1, self.ncomponents())).unwrap());
        for factor in &self.factor_matrices {
            coef_matrix = coef_matrix * factor.t().dot(&factor.view());
        }
        coef_matrix.sum().abs().sqrt()
    }

    pub fn isequal(&self, other: &Kruskal) -> bool {
        if self.ncomponents() != other.ncomponents() || self.weights != other.weights {
            return false;
        }
        for k in 0..self.ndims() {
            if self.factor_matrices[k] != other.factor_matrices[k] {
                return false;
            }
        }
        true
    }

    pub fn arrange(&mut self, permutation: &Option<&[usize]>) {
        // Only a partial implementation of arrange to get normalize settled
        let permutation = permutation.unwrap();
        if permutation.len() == self.ncomponents() {
            let mut swap = Array::<f64, Ix1>::zeros(self.ncomponents());
            for i in 0..self.weights.len() {
                swap[i] = self.weights[permutation[i]];
            }
            self.weights = swap;
            for i in 0..self.ndims() {
                let mut swap = Array::<f64, Ix2>::zeros(self.factor_matrices[i].raw_dim());
                for j in 0..permutation.len() {
                    swap.slice_mut(s![.., j])
                        .assign(&self.factor_matrices[i].slice(s![.., permutation[j]]));
                }
                self.factor_matrices[i] = swap;
            }
            return;
        }
    }

    pub fn normalize(
        &mut self,
        weight_factor: &Option<usize>,
        sort: &Option<bool>,
        normtype: &Option<i64>,
        mode: &Option<usize>,
    ) {
        let sort = sort.unwrap_or(false);
        let normtype = normtype.unwrap_or(2);

        if !mode.is_none() {
            let mode = mode.unwrap();
            if mode < self.ndims() {
                for r in 0..self.ncomponents() {
                    let tmp = p_norm(self.factor_matrices[mode].slice(s![.., r]), normtype);
                    if tmp > 0.0 {
                        self.factor_matrices[mode]
                            .slice_mut(s![.., r])
                            .map_inplace(|val| *val *= 1.0 / tmp);
                    }
                    self.weights[r] *= tmp;
                }
                return;
            } else {
                panic!("Parameter single_factor is invalid; index must be an int in range of number of dimensions");
            }
        }
        for mode in 0..self.ndims() {
            for r in 0..self.ncomponents() {
                let tmp = p_norm(self.factor_matrices[mode].slice(s![.., r]), normtype);
                if tmp > 0.0 {
                    self.factor_matrices[mode]
                        .slice_mut(s![.., r])
                        .map_inplace(|val| *val *= 1.0 / tmp);
                }
                self.weights[r] *= tmp;
            }
        }

        // Check that all weights are positive, flip sign of columns in first factor matrix if negative weight found
        for i in 0..self.ncomponents() {
            if self.weights[i] < 0.0 {
                self.weights[i] *= -1.0;
                self.factor_matrices[0]
                    .slice_mut(s![.., i])
                    .map_inplace(|val| *val *= -1.0);
            }
        }

        // TODO how to handle 'all'
        // Absorb weight into factors
        if weight_factor.is_some() && false {
            // All factors
            let d =
                Array2::from_diag(&self.weights.mapv(|val| val.powf(1.0 / self.ndims() as f64)));
            for i in 0..self.ndims() {
                self.factor_matrices[i] = self.factor_matrices[i].dot(&d);
            }
            self.weights.fill(1.0);
        } else if weight_factor.is_some() && weight_factor.unwrap() <= self.ndims() {
            // Single factor
            let idx = weight_factor.unwrap();
            let d = Array2::from_diag(&self.weights);
            self.factor_matrices[idx] = self.factor_matrices[idx].dot(&d);
            self.weights.fill(1.0);
        }

        if sort && self.ncomponents() > 1 {
            // This needs arrange :(
            let mut p = (0..self.weights.len()).collect::<Vec<_>>();
            p.sort_by(|&a_idx, &b_idx| self.weights[a_idx].total_cmp(&self.weights[b_idx]));
            self.arrange(&Some(&p));
        }
    }

    pub fn fixsigns(&mut self, other: &Option<Kruskal>) -> &Self {
        match other {
            None => {
                self.fixsigns_none();
                self
            }
            Some(tensor) => {
                self.fixsigns_other(tensor);
                self
            }
        }
    }

    fn fixsigns_none(&mut self) {
        for r in 0..self.ncomponents() {
            let mut negidx = Vec::<usize>::with_capacity(self.ndims());
            for n in 0..self.ndims() {
                let max_abs: f64 = max_abs(self.factor_matrices[n].slice(s![.., r]));
                if sign(max_abs) == -1.0 {
                    negidx.push(n);
                }
            }
            let nflip = 2 * (negidx.len() / 2);

            for i in 0..nflip {
                let n = negidx[i];
                self.factor_matrices[n]
                    .slice_mut(s![.., r])
                    .map_inplace(|val| *val *= -1.0);
            }
        }
    }

    fn fixsigns_other(&mut self, other: &Kruskal) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_kruskal_tensor() {
        let tensor = Kruskal::new();
        assert!(tensor.weights.is_empty());
        assert!(tensor.factor_matrices.is_empty());
    }

    #[test]
    fn kruskal_tensor_from_data() {
        let weights = Array::<f64, Ix1>::zeros((2,));
        let factors = vec![
            Array::<f64, Ix2>::zeros((2, 2)),
            Array::<f64, Ix2>::zeros((3, 2)),
        ];
        let tensor = Kruskal::from_data(&weights, &factors);
        assert!(!tensor.weights.is_empty());
        assert!(!tensor.factor_matrices.is_empty());
    }

    #[test]
    #[should_panic]
    fn kruskal_tensor_from_bad_data() {
        let weights = Array::<f64, Ix1>::zeros((2,));
        let factors = vec![Array::<f64, Ix2>::zeros((2, 1))];
        Kruskal::from_data(&weights, &factors);
    }

    #[test]
    fn kruskal_tensor_from_factors() {
        let factors = vec![
            Array::<f64, Ix2>::zeros((2, 2)),
            Array::<f64, Ix2>::zeros((3, 2)),
        ];
        let tensor = Kruskal::from_factor_matrices(&factors);
        assert!(!tensor.weights.is_empty());
        assert!(!tensor.factor_matrices.is_empty());
    }

    #[test]
    fn ndims() {
        let tensor = Kruskal::new();
        assert!(tensor.ndims() == 0);

        let weights = Array::<f64, Ix1>::zeros((2,));
        let factors = vec![
            Array::<f64, Ix2>::zeros((2, 2)),
            Array::<f64, Ix2>::zeros((3, 2)),
        ];
        let tensor = Kruskal::from_data(&weights, &factors);
        assert!(tensor.ndims() == factors.len());
    }

    #[test]
    fn ncomponents() {
        let tensor = Kruskal::new();
        assert!(tensor.ncomponents() == 0);

        let weights = Array::<f64, Ix1>::zeros((2,));
        let factors = vec![
            Array::<f64, Ix2>::zeros((2, 2)),
            Array::<f64, Ix2>::zeros((3, 2)),
        ];
        let tensor = Kruskal::from_data(&weights, &factors);
        assert!(tensor.ncomponents() == weights.len());
    }

    #[test]
    fn norm() {
        let factors = vec![
            Array::<f64, Ix2>::zeros((2, 2)),
            Array::<f64, Ix2>::zeros((3, 2)),
        ];
        let tensor = Kruskal::from_factor_matrices(&factors);
        let norm = tensor.norm();
        // TODO check if there is a simple approx without adding dependency
        assert!((norm - 0.0).abs() <= 1e-8);

        let factors = vec![
            Array::<f64, Ix2>::ones((2, 2)),
            Array::<f64, Ix2>::ones((2, 2)),
        ];
        let tensor = Kruskal::from_factor_matrices(&factors);
        let norm = tensor.norm();
        assert!((norm - 4.0).abs() <= 1e-8);
    }

    #[test]
    fn isequal() {
        let weights = array![1.0, 2.0];
        let factors = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[5.0, 6.0], [7.0, 8.0]],
        ];
        let tensor_1 = Kruskal::from_data(&weights, &factors);
        let mut tensor_2 = Kruskal::from_data(&weights, &factors);
        assert!(tensor_1.isequal(&tensor_2));
        tensor_2.weights[0] = 0.0;
        // Mismatch weights
        assert!(!tensor_1.isequal(&tensor_2));
        tensor_2.weights[0] = tensor_1.weights[0];

        // Mismatch factor matrices
        tensor_2.factor_matrices[0][[0, 0]] = 0.0;
        assert!(!tensor_1.isequal(&tensor_2));
    }

    #[test]
    fn fixsigns() {
        // TODO add helper for sample 2 way kruskal
        let weights = array![1.0, 2.0];
        let factors = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[5.0, 6.0], [7.0, 8.0]],
        ];
        let mut tensor_1 = Kruskal::from_data(&weights, &factors);
        let mut tensor_2 = Kruskal::from_data(&weights, &factors);
        // TODO: Update after subsref/asgn implemented
        tensor_1.factor_matrices[0]
            .slice_mut(s![1, 1])
            .map_inplace(|val| *val *= -1.0);
        tensor_1.factor_matrices[1]
            .slice_mut(s![1, 1])
            .map_inplace(|val| *val *= -1.0);
        tensor_2.factor_matrices[0]
            .slice_mut(s![0, 1])
            .map_inplace(|val| *val *= -1.0);
        tensor_2.factor_matrices[1]
            .slice_mut(s![0, 1])
            .map_inplace(|val| *val *= -1.0);
        tensor_1.fixsigns(&None);
        assert!(tensor_1.isequal(&tensor_2));
    }

    #[test]
    fn arrange() {
        let weights = array![1.0, 2.0];
        let factors = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[5.0, 6.0], [7.0, 8.0]],
        ];
        let mut tensor_1 = Kruskal::from_data(&weights, &factors);
        let tensor_2 = Kruskal::from_data(&weights, &factors);
        tensor_1.arrange(&Some(&vec![1, 0]));
        tensor_1.arrange(&Some(&vec![1, 0]));
        assert!(tensor_1.isequal(&tensor_2));
    }

    #[test]
    fn normalize() {}
}
