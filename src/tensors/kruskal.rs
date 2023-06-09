use ndarray::{Array, Ix1, Ix2};

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
