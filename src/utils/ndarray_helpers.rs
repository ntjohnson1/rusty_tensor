use ndarray::{ArrayView, Ix1};
use ndarray_linalg::Norm;

/// Gets largest value (in aboslute terms) but return original value
/// Roughly equivalent to
/// max_index = np.argmax(np.abs(data))
/// max_value = data[max_index]
pub fn max_abs(array: ArrayView<f64, Ix1>) -> f64 {
    let max_abs_value: f64 = array.iter().fold(0.0, |val_max, &val| {
        // Might want to panic on Nan?
        if val_max.abs() > val.abs() {
            val_max
        } else {
            val
        }
    });
    max_abs_value
}

/// Equivalent to the sign function in numpy
pub fn sign(value: f64) -> f64 {
    // TODO this probably already exists somewhere
    let sign_value: f64 = match value {
        value if value < 0.0 => -1.0,
        value if value > 0.0 => 1.0,
        value if value == 0.0 => 0.0,
        _ => {
            0.0 // Figure out what to do with nans
        }
    };
    sign_value
}

/// Helper to map to the name ndarray norms
pub fn p_norm(array: ArrayView<f64, Ix1>, p: i64) -> f64 {
    // TODO explore other norm options/ inf norm representation
    match p {
        1 => array.norm_l1(),
        2 => array.norm_l2(),
        _ => panic!("unsupported norm type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn test_max_abs() {
        // Check geatest value is negative
        let value = max_abs(array![-4., 3., 2., 1.].slice(s![..]));
        assert!(value == -4.0);

        // Check geatest value is postive
        let value = max_abs(array![4., 3., 2., 1.].slice(s![..]));
        assert!(value == 4.0);
    }

    #[test]
    fn test_sign() {
        assert!(sign(-2.0) == -1.0);
        assert!(sign(3.0) == 1.0);
        assert!(sign(0.0) == 0.0);
    }
}
