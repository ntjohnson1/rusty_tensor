use either::{Left, Right};
use ndarray::{Array, Ix2};

// Again want to template this against factor type
pub fn khatrirao(factors: &[Array<f64, Ix2>], reverse: &Option<bool>) -> Array<f64, Ix2> {
    let reverse = match reverse {
        Some(x) => *x,
        None => false,
    };
    let mut factor_it = if reverse {
        Left(factors.iter().rev())
    } else {
        Right(factors.iter())
    };

    let first_idx = if reverse { 0 } else { factors.len() - 1 };

    for matrix in factor_it.clone() {
        // TODO does it make sense to restrict this to Array2 and handle it by type
        if matrix.shape().len() != 2 {
            panic!("Each factor must be a matrix");
        }
    }
    let ncol_first = factors[first_idx].shape()[1];
    for matrix in factor_it.clone() {
        if matrix.shape()[1] != ncol_first {
            panic!("All matrices must have the same number of columns.");
        }
    }

    // Now lets compute something
    let p = factor_it.next().unwrap().clone(); // Grab the first and iterate over the rest
    let mut p_remaining = p.shape().iter().product::<usize>() / ncol_first;
    let mut p = p.insert_axis(ndarray::Axis(0));
    for i in factor_it {
        let i_rows = i.shape()[0];
        let i = i.clone().insert_axis(ndarray::Axis(1));
        let i = i.into_shape((i_rows, 1, ncol_first)).unwrap();
        p = p.into_shape((1, p_remaining, ncol_first)).unwrap();
        p = i * p;
        p_remaining = p.shape().iter().product::<usize>() / ncol_first;
    }
    // TODO explore a helper for creating a Shape based on all but one dim
    p_remaining = p.shape().iter().product::<usize>() / ncol_first;
    // TODO make everything above COW array until here
    p.to_shape((p_remaining, ncol_first)).unwrap().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_vectors() {
        let factors = vec![
            Array::<f64, Ix2>::ones((3, 1)) * 2.0,
            Array::<f64, Ix2>::ones((3, 1)) * 2.0,
        ];
        let result = khatrirao(&factors, &None);
        assert!(result.shape()[0] == 9);
        for value in result {
            assert!(value == 4.0);
        }

        let result = khatrirao(&factors, &Some(true));
        assert!(result.shape()[0] == 9);
        for value in result {
            assert!(value == 4.0);
        }
    }

    #[test]
    fn row_vectors() {
        let factors = vec![
            Array::<f64, Ix2>::ones((1, 3)) * 2.0,
            Array::<f64, Ix2>::ones((1, 3)) * 2.0,
        ];
        let result = khatrirao(&factors, &None);
        assert!(result.raw_dim() == factors[0].raw_dim());
        for value in result {
            assert!(value == 4.0);
        }

        let result = khatrirao(&factors, &Some(true));
        assert!(result.raw_dim() == factors[0].raw_dim());
        for value in result {
            assert!(value == 4.0);
        }
    }
}
