use ndarray::{Array, IxDyn};
use ndarray_linalg::Norm;

#[derive(Debug)]
pub struct DenseTensor {
    // TODO probably want data trait on struct
    // for different data types
    data: Array<f64, IxDyn>,
    shape: Vec<usize>,
}

impl DenseTensor {
    pub fn new() -> Self {
        Self {
            data: Array::<f64, IxDyn>::zeros(IxDyn(&[0])),
            shape: Vec::new(),
        }
    }

    pub fn from_data(data: &Array<f64, IxDyn>, shape: &Option<Vec<usize>>) -> Self {
        match shape {
            None => Self {
                data: data.clone(),
                shape: data.shape().to_vec(),
            },
            Some(value) => {
                // Placeholder
                Self {
                    data: data.clone().into_shape(value.clone()).unwrap(),
                    shape: value.clone(),
                }
            }
        }
    }

    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    pub fn innerprod(&self, other: &DenseTensor) -> f64 {
        // TODO check shape
        // TODO This makes a flat copy, check if we can do it with view
        let flat_self = Array::from_iter(self.data.iter().cloned());
        let flat_other = Array::from_iter(other.data.iter().cloned());
        flat_self.dot(&flat_other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dense_tensor() {
        let tensor = DenseTensor::new();
        assert!(tensor.shape.is_empty());
        assert!(tensor.data.is_empty());
    }

    #[test]
    fn dense_tensor_from_data() {
        let array = Array::<f64, IxDyn>::zeros(IxDyn(&[3, 3]));
        let tensor = DenseTensor::from_data(&array, &None);
        assert!(!tensor.shape.is_empty());
        assert!(!tensor.data.is_empty());

        let array = Array::<f64, IxDyn>::zeros(IxDyn(&[3, 3]));
        let shape: Vec<usize> = vec![9, 1];
        let tensor = DenseTensor::from_data(&array, &Some(shape));
        assert!(!tensor.shape.is_empty());
        assert!(!tensor.data.is_empty());
    }

    #[test]
    fn ndims() {
        let tensor = DenseTensor::new();
        assert!(tensor.ndims() == 0);

        let array = Array::<f64, IxDyn>::zeros(IxDyn(&[3, 3]));
        let tensor = DenseTensor::from_data(&array, &None);
        assert!(tensor.ndims() == 2);
    }

    #[test]
    fn norm() {
        let tensor = DenseTensor::new();
        assert!(tensor.norm() == 0.0);

        let array = Array::<f64, IxDyn>::ones(IxDyn(&[3, 3]));
        let tensor = DenseTensor::from_data(&array, &None);
        assert!(tensor.norm() == 3.0);
    }

    #[test]
    fn innerprod() {
        let tensor = DenseTensor::new();
        assert!(tensor.innerprod(&tensor) == 0.0);

        let array = Array::<f64, IxDyn>::ones(IxDyn(&[3, 3, 3]));
        let tensor = DenseTensor::from_data(&array, &None);
        assert!(tensor.innerprod(&tensor) == 27.0);
    }
}
