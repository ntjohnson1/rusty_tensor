use either::{Left, Right};
use ndarray::{s, Array, Array3, Ix2, Order, ShapeBuilder};

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
    let data = factor_it.next().unwrap(); // Grab the first and iterate over the rest
    let mut p_remaining = data.shape().iter().product::<usize>() / ncol_first;
    // This is currently a work around to insert an axis and for fortran ordering
    let mut p: Array3<f64> = Array3::<f64>::zeros((1, p_remaining, ncol_first).f());
    p.slice_mut(s![0, .., ..]).assign(data);
    for i in factor_it {
        let i_rows = i.shape()[0];
        let mut i_ = Array3::<f64>::zeros((i_rows, 1, ncol_first).f());
        i_.slice_mut(s![.., 0, ..]).assign(i);
        //let i = i.into_shape((i_rows, 1, ncol_first)).unwrap();
        p = p.into_shape((1, p_remaining, ncol_first)).unwrap();
        p = i_ * p;
        p_remaining = p.shape().iter().product::<usize>() / ncol_first;
    }
    // TODO explore a helper for creating a Shape based on all but one dim
    p_remaining = p.shape().iter().product::<usize>() / ncol_first;
    // TODO make everything above COW array until here
    p.to_shape(((p_remaining, ncol_first), Order::ColumnMajor))
        .unwrap()
        .to_owned()
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
