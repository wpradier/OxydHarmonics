use rand::{Rng, distributions::Uniform};
use libm;

#[derive(Debug)]
pub struct LinearRegressionModel {
    weights: Vec<f64>
}


pub fn create(len: usize) -> LinearRegressionModel {
    let mut rng = rand::thread_rng();
    let range = Uniform::from((-1.)..(1.));
    let vals: Vec<f64> = (0..len).map(|_| rng.sample(&range)).collect();

    LinearRegressionModel {
        weights: vals
    }
}

pub fn train(model: &mut LinearRegressionModel,
             x_train: Vec<Vec<f64>>,
             y_train: Vec<f64>,
             alpha: f64,
             epochs: u32,
             is_classification: bool) {
    println!("Weights before train: {:?}", model.weights);


    let mut rng = rand::thread_rng();

    for _ in 0..epochs {
        for _ in 0..x_train.len() {
            let k = rng.gen_range(0..x_train.len());
            let Xk = x_train.get(k).expect("Problem while trying to access input data.");
            let mut Xk_with_bias = Xk.clone();
            Xk_with_bias.push(1.);

            let prediction = predict(model, Xk, is_classification);
            let expected = y_train.get(k).expect("Problem while trying to access output data.");
            for (wi, w) in (0..model.weights.len()).zip(model.weights.iter_mut()) {
                *w += alpha * (expected - prediction) * Xk_with_bias.get(wi).expect("Sample size does not correspond to weights size.");
            }
        }
    }

    println!("Weights after train: {:?}", model.weights);
}

pub fn predict(model: &LinearRegressionModel, sample: &Vec<f64>, is_classification: bool) -> f64 {
    let mut sum = 0.;

    let mut sample_with_bias = sample.clone();
    sample_with_bias.push(1.);

    for (w, x) in model.weights.iter().zip(sample_with_bias.iter()) {
        sum += w * x;
    }

    match is_classification {
        true => { libm::tanh(sum) },
        false => { sum }
    }
}

#[cfg(test)]
mod tests {
    use crate::models::linear::create;

    #[test]
    fn sss() {
    }
}