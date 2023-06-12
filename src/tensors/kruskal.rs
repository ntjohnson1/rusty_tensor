use crate::utils::ndarray_helpers::{max_abs, sign};
use ndarray::{s, Array, Ix1, Ix2};

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
}
