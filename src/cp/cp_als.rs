use crate::tensors::dense::Dense;
use crate::tensors::kruskal::Kruskal;

//TODO add builder pattern here for argument simplicity/explicitness

pub fn cp_als(
    input_tensor: &Dense,
    rank: usize,
    stoptol: Option<f64>,
    maxiters: Option<usize>,
    dimorder: Option<&[usize]>,
    init: Option<&Kruskal>,
    printitn: Option<usize>,
    fixsigns: Option<bool>,
) -> Kruskal {
    Kruskal::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_interface() {
        let tensor = Dense::new();
        cp_als(&tensor, 1, None, None, None, None, None, None);
    }
}
