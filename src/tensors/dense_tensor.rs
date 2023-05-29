use ndarray::{Array, IxDyn, ShapeBuilder};

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
}
