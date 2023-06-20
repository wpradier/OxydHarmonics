use rand::{Rng, distributions::Uniform};

#[derive(Debug)]
pub struct LinearRegressionModel {
    len: usize,
    weights: Vec<f64>
}


pub fn create(len: usize) -> LinearRegressionModel {
    let mut rng = rand::thread_rng();
    let range = Uniform::from((0.)..(1.));
    let vals: Vec<f64> = (0..len).map(|_| rng.sample(&range)).collect();

    LinearRegressionModel {
        len,
        weights: vals
    }
}

#[cfg(test)]
mod tests {
    use crate::models::linear::create;

    #[test]
    fn sss() {
        let model = create(4);

        println!("model: {:?}", model);
    }
}