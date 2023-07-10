use rand::{Rng, distributions::Uniform};

pub fn initialise_weight() -> f64 {
    let mut rng = rand::thread_rng();
    let range = Uniform::from((-1.)..(1.));

    rng.sample(&range)
}

pub fn initialize_weights(len: usize) -> Vec<f64> {
    return (0..len).map(|_| initialise_weight()).collect();
}