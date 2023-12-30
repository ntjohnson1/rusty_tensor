use ndarray::{Array, Axis, Ix1, Ix2};
use ndarray_linalg::Norm;

#[derive(Debug)]
pub struct Sparse {
    pub subs: Array<usize, Ix2>,
    pub vals: Array<f64, Ix1>,
    pub shape: Vec<usize>,
}

impl Default for Sparse {
    fn default() -> Self {
        Self::new()
    }
}

impl Sparse {
    pub fn new() -> Self {
        Self {
            subs: Array::<usize, Ix2>::zeros(Ix2(0, 0)),
            vals: Array::<f64, Ix1>::zeros(Ix1(0)),
            shape: Vec::new(),
        }
    }

    pub fn from_data(
        subs: &Array<usize, Ix2>,
        vals: &Array<f64, Ix1>,
        shape: &Option<Vec<usize>>,
    ) -> Self {
        let subs_shape = subs
            .map_axis(Axis(0), |view| *view.iter().max().unwrap() + 1)
            .to_vec();
        let shape = shape.clone().unwrap_or(subs_shape.clone());
        if subs.shape()[1] != shape.len() || subs_shape.iter().zip(&shape).any(|(a, b)| a > b) {
            panic!(
                "Provided shape {:?} was incorrect to fit all subscripts; max subscripts are {:?}",
                shape, subs_shape
            );
        }
        Self {
            subs: subs.clone(),
            vals: vals.clone(),
            shape,
        }
    }

    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    pub fn norm(&self) -> f64 {
        self.vals.norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn get_sparse() -> Sparse {
        let subs = array![[1, 3, 0], [2, 4, 0]];
        let vals = array![3., 4.];
        Sparse::from_data(&subs, &vals, &None)
    }

    #[test]
    fn empty_sparse_tensor() {
        let tensor = Sparse::new();
        assert!(tensor.shape.is_empty());
        assert!(tensor.subs.is_empty());
        assert!(tensor.vals.is_empty());

        let tensor = Sparse::default();
        assert!(tensor.shape.is_empty());
        assert!(tensor.subs.is_empty());
        assert!(tensor.vals.is_empty());
    }

    #[test]
    fn from_data_smoke() {
        let subs = array![[1, 3, 0], [2, 4, 0]];
        let vals = array![1., 1.];
        let tensor = Sparse::from_data(&subs, &vals, &None);
        let expected_shape: Vec<usize> = vec![3, 5, 1];
        assert!(tensor.subs.abs_diff_eq(&subs, 1));
        assert!(tensor.vals.abs_diff_eq(&vals, 1e-8));
        assert!(tensor.shape.iter().eq(&expected_shape));
    }

    #[should_panic]
    #[test]
    fn constructor_incorrect_shape() {
        let subs = array![[1, 3, 0], [2, 4, 0]];
        let vals = array![1., 1.];
        let shape = vec![2, 5, 0];
        Sparse::from_data(&subs, &vals, &Some(shape));
    }

    #[test]
    fn ndims() {
        let tensor = Sparse::new();
        assert!(tensor.ndims() == 0);

        let tensor = get_sparse();
        assert!(tensor.ndims() == 3);
    }

    #[test]
    fn norm() {
        let tensor = Sparse::new();
        assert!(tensor.norm() == 0.0);

        let tensor = get_sparse();
        assert!(tensor.norm() == 5.0);
    }
}
