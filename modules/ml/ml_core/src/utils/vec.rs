use rand::{Rng, distributions::Uniform};

pub fn initialize_weights(len: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let range = Uniform::from((-1.)..(1.));
    return (0..len).map(|_| rng.sample(&range)).collect();
}