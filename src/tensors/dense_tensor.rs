use crate::utils::khatrirao::khatrirao;
use ndarray::{s, Array, Ix2, IxDyn, Order};
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

    pub fn mttkrp(&self, factors: &[Array<f64, Ix2>], n: usize) -> Array<f64, Ix2> {
        if self.ndims() < 2 {
            panic!("MTTKRP is invalid for tensors with fewer than 2 dimensions");
        }

        if factors.len() != self.ndims() {
            panic!("Second argument contains the wrong number of arrays");
        }

        let range = if n == 0 {
            factors[1].shape()[1]
        } else {
            factors[0].shape()[1]
        };

        // Check dimensions match
        for i in 0..self.ndims() {
            if i == n {
                continue;
            }
            if factors[i].shape()[0] != self.shape[i] {
                panic!("Entry {i} of list of arrays is wrong size");
            }
        }

        let szl: usize = self.shape[0..n].iter().product();
        let szr: usize = self.shape[n + 1..].iter().product();
        let szn = self.shape[n];

        if n == 0 {
            let ur = khatrirao(&factors[1..self.ndims()], &Some(true));
            let y = self
                .data
                .to_shape(((szn, szr), Order::ColumnMajor))
                .unwrap();
            return y.dot(&ur).to_owned();
        } else if n == self.ndims() - 1 {
            let ul = khatrirao(&factors[0..self.ndims() - 1], &Some(true));
            let y = self
                .data
                .to_shape(((szl, szn), Order::ColumnMajor))
                .unwrap();
            return y.t().dot(&ul).to_owned();
        } else {
            let ul = khatrirao(&factors[n + 1..], &Some(true));
            let ur = khatrirao(&factors[0..n], &Some(true))
                .to_shape(((szl, 1, range), Order::ColumnMajor))
                .unwrap()
                .to_owned();
            let y_leftover = self.data.shape().iter().product::<usize>() / szr;
            let mut y = self
                .data
                .to_shape(((y_leftover, szr), Order::ColumnMajor))
                .unwrap();
            y = y.dot(&ul).into();
            let y = y.to_shape(((szl, szn, range), Order::ColumnMajor)).unwrap();
            let mut v = Array::<f64, Ix2>::ones((szn, range));
            // TODO: The remove axis is required because the result is a column
            // check if we can take a column slice of v ([:, [r]]) in numpy
            for r in 0..range {
                v.slice_mut(s![.., r]).assign(
                    &y.slice(s![.., .., r])
                        .t()
                        .dot(&ur.slice(s![.., .., r]))
                        .remove_axis(ndarray::Axis(1)),
                );
            }
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    #[test]
    fn mttkrp_two_way() {
        let array = Array::<f64, IxDyn>::ones(IxDyn(&[2, 3, 4]));
        let tensor = DenseTensor::from_data(&array, &None);
        let factors = vec![
            array![[1.0, 3.0], [2.0, 4.0]],
            array![[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]],
            array![[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]],
        ];
        // Division is unpacking the weight from pyttb ktensor version
        let m0 = array![[1800.0, 3564.0], [1800.0, 3564.0]] / 2.0;
        let m1 = array![[300.0, 924.0], [300.0, 924.0], [300.0, 924.0]];
        let m2 = array![
            [108.0, 378.0],
            [108.0, 378.0],
            [108.0, 378.0],
            [108.0, 378.0]
        ];

        let result = tensor.mttkrp(&factors, 0);
        assert!(result.abs_diff_eq(&m0, 1e-8));

        let result = tensor.mttkrp(&factors, 1);
        print!("Result: {:?} and correct {:?}", result, m0);
        result.abs_diff_eq(&m1, 1e-8);

        let result = tensor.mttkrp(&factors, 2);
        result.abs_diff_eq(&m2, 1e-8);
    }
}
