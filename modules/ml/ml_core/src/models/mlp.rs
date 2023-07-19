use std::cmp::max;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use chrono::{DateTime, Local};
use libm::fmax;
use rand::Rng;
use tensorboard_rs::summary_writer::SummaryWriter;
use crate::utils::vec::initialise_weight;
use bincode::{serialize, deserialize};
use serde::{Deserialize, Serialize};
use serde::ser::SerializeStruct;

pub struct MultilayerPerceptron {
    structure: Vec<usize>,
    layers: usize,
    weights: Vec<Vec<Vec<f64>>>,
    neurons_outputs: Vec<Vec<f64>>,
    gradients: Vec<Vec<f64>>
}

impl Serialize for MultilayerPerceptron {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize each field of the struct manually
        let mut state = serializer.serialize_struct("MultilayerPerceptron", 5)?;

        // Serialize the fields one by one
        state.serialize_field("structure", &self.structure)?;
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("weights", &self.weights)?;
        state.serialize_field("neurons_outputs", &self.neurons_outputs)?;
        state.serialize_field("gradients", &self.gradients)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for MultilayerPerceptron {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize each field of the struct manually
        struct MultilayerPerceptronVisitor;

        impl<'de> serde::de::Visitor<'de> for MultilayerPerceptronVisitor {
            type Value = MultilayerPerceptron;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct MultilayerPerceptron")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // Deserialize each field one by one
                let val_d = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let val_l = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let val_w = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let val_x = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let val_deltas = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;

                Ok(MultilayerPerceptron {
                    structure: val_d,
                    layers: val_l,
                    weights: val_w,
                    neurons_outputs: val_x,
                    gradients: val_deltas
                })
            }
        }

        deserializer.deserialize_struct("MultilayerPerceptron", &["structure", "layers", "weights", "neurons_outputs", "gradients"], MultilayerPerceptronVisitor)
    }
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
    is_classification: bool,
    train_name: &str
) {
    let current_time: DateTime<Local> = Local::now();
    let formatted_time = current_time.format("%Y-%m-%d_%H-%M-%S").to_string();
    let log_path = format!("{}_{}", train_name, formatted_time);
    let mut rng = rand::thread_rng();

    for epoch in 0..epochs {
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

        if is_classification {
            write_classif_stats(
                model,
                x_train,
                y_train,
                epoch,
                log_path.to_string()
            )
        } else {
            write_regression_stats(
                model,
                x_train,
                y_train,
                epoch,
                log_path.to_string()
            )
        }
    }
}

pub fn save(model: &MultilayerPerceptron, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let encoded: Vec<u8> = serialize(model)?;
    let mut file = File::create(filename)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load(filename: &str) -> Result<MultilayerPerceptron, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;
    let model: MultilayerPerceptron = deserialize(&encoded)?;
    Ok(model)
}


fn write_classif_stats(
    model: &mut MultilayerPerceptron,
    samples: &Vec<Vec<f64>>,
    expected_values: &Vec<Vec<f64>>,
    epoch: u32,
    log_path: String
) {
    let mut writer = SummaryWriter::new(&"./logdir");
    let mut loss_map = HashMap::new();
    let mut acc_map = HashMap::new();

    let predictions: Vec<Vec<f64>> = samples
        .iter()
        .map(|input| {
            let input_k = input.as_ref();
            predict(model, input_k, true)
        })
        .collect::<Vec<Vec<f64>>>();

    let loss = log_loss(
        expected_values,
        &predictions
    );

    let accuracy = classif_accuracy(
        expected_values,
        &predictions
    );

    loss_map.insert(log_path.to_string(), loss as f32);
    acc_map.insert(log_path.to_string(), accuracy as f32);

    writer.add_scalars("epoch_loss", &loss_map, epoch as usize + 1);
    writer.add_scalars("epoch_accuracy", &acc_map, epoch as usize + 1);
}

fn log_loss(
    expected_values: &Vec<Vec<f64>>,
    predictions: &Vec<Vec<f64>>
) -> f64 {
    let mut loss = 0.;
    let mut total_values: f64 = 0.;

    for (prediction, expected) in predictions.iter()
    .zip(expected_values.iter()) {
        for (predict_value, expected_value) in prediction.iter()
            .zip(expected.iter()) {
            total_values += 1.;
            /* My adapted logloss for tanh activation */
            let a = -1. * (expected_value - 1.) / 2.;
            let b = (expected_value + 1.) / 2.;

            loss += a * (-1. * (predict_value - 1.) / 2.).ln()
                + b * ((predict_value + 1.) / 2.).ln()
        }
    }
    loss = -loss / total_values;

    loss
}

fn classif_accuracy(
    expected_values: &Vec<Vec<f64>>,
    predictions: &Vec<Vec<f64>>
) -> f64 {
    let mut accurate_predictions = 0;
    for (prediction, expected) in predictions.iter()
    .zip(expected_values.iter()) {
        if *prediction == *expected {
            accurate_predictions += 1
        }
    }

    return f64::try_from(accurate_predictions).unwrap() / f64::try_from(predictions.len() as i32).unwrap();
}

fn write_regression_stats(
    model: &mut MultilayerPerceptron,
    samples: &Vec<Vec<f64>>,
    expected_values: &Vec<Vec<f64>>,
    epoch: u32,
    log_path: String
) {
    let mut writer = SummaryWriter::new(&"./logdir");
    let mut mse_map = HashMap::new();

    let predictions: Vec<Vec<f64>> = samples
        .iter()
        .map(|input| {
            let input_k = input.as_ref();

            predict(model, input_k, false)
        })
        .collect::<Vec<Vec<f64>>>();

    let mse = mean_squared_error(expected_values, &predictions);

    mse_map.insert(log_path.to_string(), mse as f32);

    writer.add_scalars("mean_squared_error", &mse_map, epoch as usize + 1);
}

fn mean_squared_error(
    expected_values: &Vec<Vec<f64>>,
    predictions: &Vec<Vec<f64>>
) -> f64 {
    let mut total_values = 0.;
    let mut mse = 0.;
    for (expected, prediction) in expected_values.iter()
        .zip(predictions.iter()) {
        for (expected_val, pred_val) in expected.iter()
            .zip(prediction.iter()) {
            total_values += 1.;
            mse += (expected_val - pred_val).powi(2);
        }
    }
    mse = mse / total_values;

    mse
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
            true,
            "test"
        );

        println!("AFTER TRAIN");
        println!("WEIGHTS: {:?}", mlp.weights);
        for samp in &sample_input {
            let res = predict(&mut mlp, samp, true);
            println!("SAMPLE {:?}: {:?}", samp, res);
        }

    }
}