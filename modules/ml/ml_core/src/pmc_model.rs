use std::collections::HashMap;
use std::ops::Index;
use rand::Rng;
use tensorboard_rs::summary_writer::SummaryWriter;
use chrono::{DateTime, Local};
use std::{f64::consts::E, usize};
use serde::{Serialize, Deserialize};
use serde::ser::SerializeStruct;
use rand_distr::StandardNormal;
use std::fs::File;
use std::io::Write;
use std::io::Read;
use bincode::{serialize, deserialize};




pub struct PmcModel {
    pub d: Vec<usize>,
    pub L: usize,
    pub W: Vec<Vec<Vec<f64>>>,
    pub X: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
}

impl Serialize for PmcModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize each field of the struct manually
        let mut state = serializer.serialize_struct("PmcModel", 5)?;

        // Serialize the fields one by one
        state.serialize_field("d", &self.d)?;
        state.serialize_field("L", &self.L)?;
        state.serialize_field("W", &self.W)?;
        state.serialize_field("X", &self.X)?;
        state.serialize_field("deltas", &self.deltas)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for PmcModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize each field of the struct manually
        struct PmcModelVisitor;

        impl<'de> serde::de::Visitor<'de> for PmcModelVisitor {
            type Value = PmcModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct PmcModel")
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

                Ok(PmcModel { d: val_d, L: val_l, W: val_w, X: val_x, deltas: val_deltas })
            }
        }

        deserializer.deserialize_struct("PmcModel", &["d", "L", "W", "X", "deltas"], PmcModelVisitor)
    }
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
            /* let mut rng = rand::thread_rng();
            let dist = Uniform::new_inclusive(-1.0, 1.0);
            let random_number = rng.sample(dist); */

            W[l].push(Vec::new());
            
            for j in 0..npl[l] + 1 {
                let mut rng = rand::thread_rng();
                let dist = StandardNormal;
                let random_number = rng.sample(dist);
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
    println!("w: {:?}", W);
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

pub fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&x| (x - max_score).exp()).collect();
    let sum_exp_scores: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|&x| x / sum_exp_scores).collect()
}


pub fn propagate_pmc<T>(model: &mut PmcModel, inputs: Vec<T>, is_classification: bool)
where
    T: AsRef<[f64]> + Clone,
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

pub fn calculate_loss(predictions: &[f64], labels: &[f64], is_classification: bool) -> f64 {
    let num_samples = predictions.len();
    let mut loss = 0.0;

    if is_classification {
        // Classification loss (cross-entropy)
        for (prediction, label) in predictions.iter().zip(labels.iter()) {
            let p = *prediction;
            let y = *label;
            if p > 0.0 && p < 1.0 {
                loss += -y * p.ln() - (1.0 - y) * (1.0 - p).ln();
            }
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

pub fn train_pmc<T, F>(
    mut model: &mut PmcModel,
    features: Vec<F>,
    labels: Vec<T>,
    is_classification: bool,
    num_iter: i32,
    alpha: f64,
) -> &mut PmcModel
where
    T: Index<usize, Output = f64> + AsRef<[f64]>,
    F: AsRef<[f64]> + Clone,
{
    let mut writer = SummaryWriter::new(&"./logdir");
    let current_time: DateTime<Local> = Local::now();
    let formatted_time = current_time.format("%Y-%m-%d_%H-%M-%S").to_string();
    let log_path = format!("data_{}", formatted_time);

    let mut map = HashMap::new();

    for iter in 0..num_iter {
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

        if iter % 100 == 0 {
            let predictions = features
                .iter()
                .map(|input| {
                    let input_k = input.as_ref();
                    predict_pmc(&mut model, input_k, is_classification)[0]
                })
                .collect::<Vec<f64>>();

            // Exclude samples with NaN predictions from the calculation of loss
            let valid_predictions: Vec<f64> = predictions
                .iter()
                .zip(labels.iter())
                .filter(|(prediction, _)| prediction.is_finite())
                .map(|(prediction, _)| *prediction)
                .collect();

            let loss = calculate_loss(
                &valid_predictions,
                &labels
                    .iter()
                    .filter(|label| label[0].is_finite())
                    .map(|label| label[0])
                    .collect::<Vec<f64>>(),
                is_classification,
            );
            //println!("iter: {}, loss: {}", iter, loss);
            map.insert("loss".to_string(), loss as f32);
            writer.add_scalars(&log_path, &map, iter as usize);
        }
    }

    model
}

pub fn save_pmc(model: &PmcModel, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let encoded: Vec<u8> = serialize(model)?;
    let mut file = File::create(filename)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load_pmc(filename: &str) -> Result<PmcModel, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;
    let model: PmcModel = deserialize(&encoded)?;
    Ok(model)
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
   
   
   
        let mut model = create_pmc(vec![2,3,4 ,1]);

        //let mut model = load_pmc("filename.json").unwrap();
   
          train_pmc(
               &mut model,
               X.clone(),
               Y.clone(),
               false,
               10000,
               0.01,
           ); 
   
           for sample in &X {
               let prediction = predict_pmc(&mut model, sample.clone(), false);
               println!("{:?}",  prediction)
               }
            //save_pmc(&model, "filename.json").unwrap();
   
       }
}
}
