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

    pub fn ndims(&self) -> usize {
        self.factor_matrices.len()
    }

    pub fn ncomponents(&self) -> usize {
        self.weights.len()
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
}
