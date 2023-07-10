use rand::Rng;
use crate::utils::vec::initialise_weight;

pub struct MultilayerPerceptron {
    structure: Vec<usize>,
    layers: usize,
    weights: Vec<Vec<Vec<f64>>>,
    neurons_outputs: Vec<Vec<f64>>,
    gradients: Vec<Vec<f64>>
}

pub fn create(structure: Vec<usize>) -> MultilayerPerceptron {
    let layers = structure.len() - 1;
    let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();

    for l in 0..layers + 1 {
        weights.push(Vec::new());

        if l == 0 {
            continue;
        }

        for i in 0..structure[l - 1] + 1 {
            weights[l].push(Vec::new());
            for j in 0..structure[l] + 1 {
                weights[l][i].push(
                    match j {
                        0 => 0.,
                        _ => initialise_weight()
                    }
                )
            }
        }
    }

    let mut neurons_outputs: Vec<Vec<f64>> = Vec::new();

    for l in 0..layers + 1 {
        neurons_outputs.push(Vec::new());
        for j in 0..structure[l] + 1 {
            neurons_outputs[l].push(
                match j {
                    0 => 1.,
                    _ => 0.
                }
            )
        }
    }

    let mut gradients: Vec<Vec<f64>> = Vec::new();

    for l in 0..layers + 1 {
        gradients.push(Vec::new());
        for j in 0..structure[l] + 1 {
            gradients[l].push(0.);
        }
    }

    MultilayerPerceptron {
        structure,
        layers,
        weights,
        neurons_outputs,
        gradients
    }
}

fn propagate(model: &mut MultilayerPerceptron, sample: &Vec<f64>, is_classification: bool) {
    // Set input layer with sample values
    for j in 0..model.structure[0] {
        model.neurons_outputs[0][j + 1] = sample[j]
    }

    for l in 1..model.layers + 1 {
        for j in 1..model.structure[l] + 1 {
            let mut total = 0.;
            // j commence à 1 car le biais de la couche L ne se connecte pas à la couche précédente.
            // i commence à 0 car les neurones de la couche actuelle se connectent au biais de la couche précédente.
            for i in 0..model.structure[l - 1] + 1 {
                total += model.weights[l][i][j] * model.neurons_outputs[l - 1][i]
            }

            if l < model.layers || is_classification {
                total = libm::tanh(total);
            }

            model.neurons_outputs[l][j] = total;
        }
    }
}

pub fn predict(model: &mut MultilayerPerceptron, sample: &Vec<f64>, is_classification: bool) -> Vec<f64> {
    propagate(model, sample, is_classification);

    let mut prediction = model.neurons_outputs[model.layers].clone();
    prediction.remove(0);

    return prediction;
}

pub fn train(
    model: &mut MultilayerPerceptron,
    x_train: &Vec<Vec<f64>>,
    y_train: &Vec<Vec<f64>>,
    alpha: f64,
    epochs: u32,
    is_classification: bool
) {
    let mut rng = rand::thread_rng();

    for _ in 0..epochs {
        for _ in 0..x_train.len() {
            let k = rng.gen_range(0..x_train.len());

            let Xk = x_train.get(k).unwrap();
            let Yk = y_train.get(k).unwrap();

            propagate(model, Xk, is_classification);

            for j in 1..model.structure[model.layers] + 1 {
                model.gradients[model.layers][j] = model.neurons_outputs[model.layers][j] - Yk[j - 1];

                if is_classification {
                    model.gradients[model.layers][j] *= 1. - (model.neurons_outputs[model.layers][j].powi(2))
                }
            }

            for l in (1..model.layers + 1).rev() {
                for i in 1..model.structure[l - 1] + 1 {
                    let mut total = 0.;
                    for j in 1..model.structure[l] + 1 {
                        total += model.weights[l][i][j] * model.gradients[l][j];
                    }
                    model.gradients[l - 1][i] = (1. - model.neurons_outputs[l - 1][i].powi(2)) * total;
                }
            }

            for l in 1..model.layers + 1 {
                for i in 0..model.structure[l - 1] + 1 {
                    for j in 1..model.structure[l] + 1 {
                        model.weights[l][i][j] -= alpha * model.neurons_outputs[l - 1][i] * model.gradients[l][j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::models::mlp;
    use crate::models::mlp::{create, predict, propagate, train};

    #[test]
    fn sss() {
        let mut mlp = create(vec![2, 2, 1]);
        let sample_input = vec![vec![0.,0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        let sample_output = vec![vec![-1.], vec![1.], vec![1.], vec![-1.]];

        println!("BEFORE TRAIN");
        println!("WEIGHTS: {:?}", mlp.weights);
        for samp in &sample_input {
            let res = predict(&mut mlp, samp, true);
            println!("SAMPLE {:?}: {:?}", samp, res);
        }

        println!("training...");
        train(&mut mlp,
              &sample_input,
            &sample_output,
            0.01,
            25000,
            true
        );

        println!("AFTER TRAIN");
        println!("WEIGHTS: {:?}", mlp.weights);
        for samp in &sample_input {
            let res = predict(&mut mlp, samp, true);
            println!("SAMPLE {:?}: {:?}", samp, res);
        }

    }
}