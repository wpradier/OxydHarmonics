use rand::Rng;
use std::char::from_digit;
use std::{f64::consts::E, usize};
use ndarray::*;
use ndarray::Dim;
use std::iter::Map;



fn main() {
    let X = vec![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let Y = vec![[42.], [51.], [19.], [5.]];
    /*let X = vec![
        [1., 1.],
        [2., 3.],
        [3., 3.]
  ];
    let Y = vec![
        [1.],
        [-1.],
        [-1.]
  ];
*/

                       /* PMC */
    let mut model = create_pmc(vec![2, 5, 1]);

    model = train_pmc(
        model,
        X.clone(),
        Y,
        false,
        100000,
        0.01,
    );

    for sample in X {
        model = predict_pmc(model, sample, false);
        println!("{:?}",  model.X[model.L][1..].to_vec())
        }
}


#[allow(non_snake_case)]
#[derive(Clone)]
pub struct PmcModel {
    pub d: Vec<i32>,
    pub L: usize,
    pub W: Vec<Vec<Vec<f64>>>,
    pub X: Vec< Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
}

pub fn create_pmc(npl: Vec<i32>)  -> PmcModel{
    
    let d:Vec<i32> = npl.clone();
    let L = npl.len() - 1;
    let mut W: Vec<Vec<Vec<f64>>> = Vec::new();

    for l in 0..L + 1 {
        W.push(vec![]);
        if l == 0 {
            continue;
        }
        for i in 0..npl[l - 1] + 1 {
            let mut rng = rand::thread_rng();
           
            W[l].push(Vec::new());
            for j in 0..npl[l] + 1 {
                W[l][i as usize].push(if j == 0 { 0.0 } else {rng.gen_range(-1. ..1.) });
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
    PmcModel { d: d, L: L, W: W, X: X, deltas: deltas }
}

fn tanh(x: f64) -> f64 {
    (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
}

fn propagate_pmc(mut model :PmcModel , inputs: [f64; 2], is_classification: bool) -> PmcModel {
    for j in 0..model.d[0] {
        model.X[0][j as usize + 1] = inputs[j as usize] as f64;
    }
    for l in 1..model.L + 1 {
        for j in 1..model.d[l] + 1 {
            let mut total = 0.0;
            for i in 0..model.d[l - 1] + 1 {
                total += model.W[l][i as usize][j as usize] * model.X[l - 1][i as usize]
            }
            if l < model.L || is_classification {
                total = tanh(total);
            }
            model.X[l][j as usize] = total;
        }
    }
    model
}

fn predict_pmc(mut model : PmcModel, inputs: [f64; 2], is_classification: bool) -> PmcModel {
    model = propagate_pmc(model,inputs, is_classification);
    model
}

fn train_pmc(
    mut model : PmcModel,
    features: Vec<[f64; 2]>,
    label: Vec<[f64; 1]>,
    is_classification: bool,
    num_iter: i32,
    alpha: f64,
)-> PmcModel{
    for _ in 0..num_iter {
        let mut rng = rand::thread_rng();
        let k = rng.gen_range(0..features.len());
        let input_k = features[k];
        let y_k = label[k];

        model = propagate_pmc(model ,input_k, is_classification);

        for j in 1..model.d[model.L] + 1 {
            model.deltas[model.L][j as usize] = model.X[model.L][j as usize]   - y_k[j as usize - 1] as f64;
            if is_classification {
                model.deltas[model.L][j as usize] *= 1. - model.X[model.L][j as usize].powf(2.);
            }
        }

        for l in (1..model.L + 1).rev() {
            for i in 1..model.d[l - 1] + 1 {
                let mut total = 0.0;
                for j in 1..model.d[l] + 1 {
                    total += model.W[l][i as usize][j as usize] * model.deltas[l][j as usize];
                }
                model.deltas[l - 1][i as usize] = (1. - model.X[l - 1][i as usize].powf(2.)) * total;
            }
        }

        for l in 1..model.L + 1 {
            for i in 0..model.d[l - 1] + 1 {
                for j in 1..model.d[l] + 1 {
                    model.W[l][i as usize][j as usize] -= alpha * model.X[l - 1][i as usize] * model.deltas[l][j as usize]
                }
            }
        }
    }
    model
}
