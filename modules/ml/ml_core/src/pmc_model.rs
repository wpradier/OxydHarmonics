use std::collections::HashMap;
use std::ops::Index;
use rand::Rng;
use tensorboard_rs::summary_writer::SummaryWriter;
use rand::distributions::uniform::Uniform;
use chrono::{DateTime, Local};
use std::{f64::consts::E, usize};


pub struct PmcModel {
    pub d: Vec<usize>,
    pub L: usize,
    pub W: Vec<Vec<Vec<f64>>>,
    pub X: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
}

pub fn create_pmc(npl: Vec<usize>) -> PmcModel {
    let d: Vec<usize> = npl.clone();
    let L = npl.len() - 1;
    let mut W: Vec<Vec<Vec<f64>>> = Vec::new();

    for l in 0..L + 1 {
        W.push(vec![]);
        if l == 0 {
            continue;
        }
        for i in 0..npl[l - 1] + 1 {
            let mut rng = rand::thread_rng();
            let dist = Uniform::new_inclusive(-1.0, 1.0);
            let random_number = rng.sample(dist);

            W[l].push(Vec::new());
            for j in 0..npl[l] + 1 {
                W[l][i].push(if j == 0 { 0.0 } else { random_number });
            }
        }
    }

    let mut X: Vec<Vec<f64>> = Vec::new();
    for l in 0..L + 1 {
        X.push(Vec::new());
        for j in 0..npl[l] + 1 {
            X[l].push(if j == 0 { 1.0 } else { 0.0 });
        }
    }

    let mut deltas: Vec<Vec<f64>> = Vec::new();
    for l in 0..L + 1 {
        deltas.push(Vec::new());
        for _ in 0..npl[l] + 1 {
            deltas[l].push(0.0);
        }
    }
    PmcModel {
        d: d,
        L: L,
        W: W,
        X: X,
        deltas: deltas,
    }
}

pub fn tanh(x: f64) -> f64 {
    (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
}

pub fn propagate_pmc<T>(model: &mut PmcModel, inputs: Vec<T>, is_classification: bool) 
where
    T: AsRef<[f64]> +Clone,
{
    for input in inputs {
        let input_k = input.as_ref();
        for j in 0..model.d[0] {
            model.X[0][j + 1] = input_k[j] as f64;
        }

        for l in 1..model.L + 1 {
            for j in 1..model.d[l] + 1 {
                let mut total = 0.0;
                for i in 0..model.d[l - 1] + 1 {
                    total += model.W[l][i][j] * model.X[l - 1][i];
                }
                if l < model.L || is_classification {
                    total = tanh(total);
                }
                model.X[l][j] = total;
            }
        }
    }

    
}

pub fn predict_pmc<T>(model: &mut PmcModel, input: T, is_classification: bool) -> Vec<f64>
where
    T: AsRef<[f64]>,
{
    let input_k = input.as_ref();
    propagate_pmc(model, vec![input_k], is_classification);
    model.X[model.L][1..].to_vec()
}



pub fn calculate_accuracy(predictions: &[f64], labels: &[f64], is_classification: bool) -> f64 {
    let num_samples = predictions.len();
    let mut num_correct = 0;

    if is_classification {
        // Classification
        for (prediction, label) in predictions.iter().zip(labels.iter()) {
            let predicted_class = if *prediction >= 0.5 { 1.0 } else { 0.0 };
            if predicted_class == *label {
                num_correct += 1;
            }
        }
    } else {
        // Regression
        for (prediction, label) in predictions.iter().zip(labels.iter()) {
            let error = (prediction - label).abs();
            if error < 0.5 {
                num_correct += 1;
            }
        }
    }

    let accuracy = (num_correct as f64) / (num_samples as f64);
    accuracy
}

pub fn calculate_loss(predictions: &[f64], labels: &[f64], is_classification: bool) -> f64 {
    let num_samples = predictions.len();
    let mut loss = 0.0;
    
    if is_classification {
        // Classification loss
        for (prediction, label) in predictions.iter().zip(labels.iter()) {
            let p = *prediction;
            let y = *label;
            loss += -y * p.ln() - (1.0 - y) * (1.0 - p).ln();
        }
    
        loss / (num_samples as f64)
    } else {
        // Regression loss
        for (prediction, label) in predictions.iter().zip(labels.iter()) {
            let error = prediction - label;
            loss += error * error;
        }
    
        loss / (num_samples as f64)
    }
}

/* pub fn create_batches<T: Clone>(
    batch_size: usize,
    features: &[Vec<T>],
    labels: &[Vec<T>],
) -> Vec<(Vec<Vec<T>>, Vec<Vec<T>>)> {
    let num_samples = features.len();
    let num_batches = (num_samples + batch_size - 1) / batch_size;

    let mut batches = Vec::with_capacity(num_batches);

    let mut indices: Vec<usize> = (0..num_samples).collect();
    indices.shuffle(&mut rand::thread_rng());

    for i in 0..num_batches {
        let start_idx = i * batch_size;
        let end_idx = (start_idx + batch_size).min(num_samples);

        let mut batch_features = Vec::with_capacity(end_idx - start_idx);
        let mut batch_labels = Vec::with_capacity(end_idx - start_idx);

        for j in start_idx..end_idx {
            batch_features.push(features[indices[j]].clone());
            batch_labels.push(labels[indices[j]].clone());
        }

        batches.push((batch_features, batch_labels));
    }

    batches
} */




pub fn train_pmc<T, F>(
    mut model: &mut PmcModel,
    features: Vec<F>,
    labels: Vec<T>,
    is_classification: bool,
    num_iter: i32,
    alpha: f64,
) -> &mut PmcModel where
T: Index<usize, Output = f64> + AsRef<[f64]>,
F: AsRef<[f64]> + Clone,
    
{
    let mut writer = SummaryWriter::new(&"./logdir");
    let mut map = HashMap::new();
    let current_time: DateTime<Local> = Local::now();
    let formatted_time = current_time.format("%Y-%m-%d_%H-%M-%S").to_string();
    let log_path = format!("data_{}", formatted_time);

    //let labels_as_ref: Vec<Vec<f64>> = labels.iter().map(|label| label.as_ref().to_vec()).collect();

    for _ in 0..num_iter {
        //let batches = create_batches(batch_size, &features.iter().map(|item| item.as_ref().to_vec()).collect::<Vec<Vec<f64>>>(), &labels_as_ref);

        //for (batch_features, batch_labels) in batches {
                let mut rng = rand::thread_rng();
                let k = rng.gen_range(0..features.len());
                let input_k = &features[k];
                let y_k = &labels[k];

                propagate_pmc(&mut model, vec![input_k], is_classification);

                for j in 1..model.d[model.L] + 1 {
                    model.deltas[model.L][j] = model.X[model.L][j] - y_k[j - 1] as f64;
                    if is_classification {
                        model.deltas[model.L][j] *= 1. - model.X[model.L][j].powf(2.);
                    }
                }

                for l in (1..model.L + 1).rev() {
                    for i in 1..model.d[l - 1] + 1 {
                        let mut total = 0.0;
                        for j in 1..model.d[l] + 1 {
                            total += model.W[l][i][j] * model.deltas[l][j];
                        }
                        model.deltas[l - 1][i] = (1. - model.X[l - 1][i].powf(2.)) * total;
                    }
                }

                for l in 1..model.L + 1 {
                    for i in 0..model.d[l - 1] + 1 {
                        for j in 1..model.d[l] + 1 {
                            model.W[l][i][j] -= alpha * model.X[l - 1][i] * model.deltas[l][j];
                        }
                    }
                }
            }
            let predictions = features
            .iter()
            .map(|input| {
                let input_k = input.as_ref();
                predict_pmc(&mut model, input_k, is_classification)[0]
            })
            .collect::<Vec<f64>>();
        let accuracy = calculate_accuracy(&predictions, &labels.iter().map(|label| label[0]).collect::<Vec<f64>>(), is_classification);
        let loss = calculate_loss(&predictions, &labels.iter().map(|label| label[0]).collect::<Vec<f64>>(), is_classification);
        map.insert("accuracy".to_string(), accuracy as f32);
        map.insert("loss".to_string(), loss as f32);
        writer.add_scalars(&log_path, &map, num_iter as usize);
        model
        }

        





fn main() {
    for i in 0..=10 {
        if i ==0{
           println!("=============================================");
           println!("CLASSIFICATION");
           println!("=============================================");
       }
   
       if i == 0{
           println!("=============================================");
           println!("Classification Linear simple");
           println!("=============================================");
           let X = vec![
       [1., 1.],
       [2., 3.],
       [3., 3.]
   ];
           let Y = vec![
       [1.],
       [-1.],
       [-1.]
   ];
       println!("expected output: {:?}", Y);
   
   
   
           let mut model = create_pmc(vec![2, 1]);
   
           train_pmc(
               &mut model,
               X.clone(),
               Y.clone(),
               false,
               100000,
               0.01,
           );
   
           for sample in &X {
               let prediction = predict_pmc(&mut model, sample.clone(), false);
               println!("{:?}",  prediction)
               }
   
       }
}
}
