use rand::prelude::{SliceRandom, ThreadRng};
use rand::thread_rng;
use rand::Rng;
use std::f64::consts::PI;
use core::f64::consts::E;

fn kmeans(X: &[Vec<f64>], k: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = X.len();
    let d = X[0].len();
    let mut centers = vec![vec![0.0; d]; k];
    let mut stds = vec![0.0; k];

    // Initialisation aléatoire des centres
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    for i in 0..k {
        centers[i] = X[indices[i]].clone();
    }

    // Boucle principale du k-means
    let mut prev_centers = vec![vec![0.0; d]; k];
    while centers != prev_centers {
        prev_centers.clone_from_slice(&centers);
        let mut counts = vec![0; k];
        let mut sums = vec![vec![0.0; d]; k];
        for x in X {
            let (min_index, min_dist) = (0..k)
                .map(|i| {
                    let dist = (0..d)
                        .map(|j| (x[j] - centers[i][j]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (i, dist)
                })
                .min_by(|(_, dist1), (_, dist2)| dist1.partial_cmp(dist2).unwrap())
                .unwrap();
            for j in 0..d {
                sums[min_index][j] += x[j];
            }
            counts[min_index] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..d {
                    centers[i][j] = sums[i][j] / counts[i] as f64;
                }
            }
        }
    }

    // Calcul des écarts types
    let mut std_sum = 0.0;
    let mut count = 0;
    for x in X {
        let min_dist = (0..k)
            .map(|i| {
                (0..d)
                    .map(|j| (x[j] - centers[i][j]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .min_by(|dist1, dist2| dist1.partial_cmp(dist2).unwrap())
            .unwrap();
        std_sum += min_dist;
        count += 1;
    }
    let mean_std = std_sum / count as f64;
    stds.iter_mut().for_each(|std| *std = mean_std);

    (centers, stds)
}

fn matrix_multiply(m1: &[Vec<f64>], m2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m1.len();
    let cols = m2[0].len();
    let n = m1[0].len();
    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..n {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    result
}

fn matrix_multiply_vector(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let rows = m.len();
    let cols = v.len();
    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += m[i][j] * v[j];
        }
    }
    result
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = m[i][j];
        }
    }
    result
}

fn matrix_inverse(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a = m.to_vec();
    let mut e = vec![vec![0.0; n]; n];
    for i in 0..n {
        e[i][i] = 1.0;
    }

    for i in 0..n {
        let mut pivot = a[i][i];
        if pivot == 0.0 {
            for j in (i + 1)..n {
                if a[j][i] != 0.0 {
                    for k in 0..n {
                        a[i][k] += a[j][k];
                        e[i][k] += e[j][k];
                    }
                    pivot = a[i][i];
                    break;
                }
            }
            if pivot == 0.0 {
                return None; // La matrice n'est pas inversible
            }
        }

        for j in 0..n {
            a[i][j] /= pivot;
            e[i][j] /= pivot;
        }

        for j in 0..n {
            if j != i {
                let ratio = a[j][i];
                for k in 0..n {
                    a[j][k] -= ratio * a[i][k];
                    e[j][k] -= ratio * e[i][k];
                }
            }
        }
    }

    Some(e)
}

fn rbf(x: f64, c: f64, s: f64) -> f64 {
    (-1.0 / (2.0 * s.powi(2)) * (x - c).powi(2)).exp()
}

pub struct RBFNet {
    pub k: i32,
    pub infer_stds: bool,
    pub centers: Vec<Vec<f64>>,
    pub stds: Vec<f64>,
    pub w: Vec<f64>,
    pub b: f64,
}

pub fn create(layers: &[usize]) -> RBFNet {
    let k = layers.len() - 1;
    let centers = Vec::new();
    let stds = Vec::new();
    let w = generate_random_weights(k);
    let b = 0.0;

    RBFNet {
        k: k as i32,
        infer_stds: true,
        centers,
        stds,
        w,
        b,
    }
}

pub fn generate_random_weights(k: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..k)
        .map(|_| rng.gen_range(-1. .. 1.))
        .collect::<Vec<f64>>()
}

pub fn train(model: &mut RBFNet, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
    let n = X.len();
    let d = X[0].len();
    let (centers, stds) = if model.infer_stds {
        kmeans(X, model.k as usize)
    } else {
        (model.centers.clone(), model.stds.clone())
    };

    // Vérification de la taille des vecteurs dans centers
    for center in &centers {
        if center.len() != d {
            panic!("Les vecteurs dans centers doivent avoir la même taille que les vecteurs dans X.");
        }
    }

    model.centers = centers;
    model.stds = stds;

    // Calcul de la matrice 𝜙 et de 𝑊
    let mut phi = vec![vec![0.0; model.k as usize]; n];
    for i in 0..n {
        for j in 0..model.k {
            let dist = (0..d)
                .map(|k| (X[i][k] - model.centers[j as usize][k]).powi(2))
                .sum::<f64>()
                .sqrt();
            phi[i][j as usize] = rbf(dist, 0.0, model.stds[j as usize]);
        }
    }

    let phi_transpose = transpose(&phi);
    let phi_transpose_phi = matrix_multiply(&phi_transpose, &phi);
    let phi_transpose_y = matrix_multiply_vector(&phi_transpose, &y);
    let inv_phi_transpose_phi = matrix_inverse(&phi_transpose_phi).expect("Impossible de calculer l'inverse de (𝜙𝑇𝜙)");
    model.w = matrix_multiply_vector(&inv_phi_transpose_phi, &phi_transpose_y);

    model.b = 0.0;

    let mut rng = thread_rng();
    for _ in 0..epochs {
        let random_indices: Vec<usize> = (0..n).collect();
        let shuffled_indices: Vec<usize> = random_indices.choose_multiple(&mut rng, n).cloned().collect();
        for &i in &shuffled_indices {
            
            let mut a = vec![0.0; model.k as usize];
            for (j, c) in model.centers.iter().enumerate() {
                let dist = (0..d)
                    .map(|k| (X[i][k] - c[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                a[j] = rbf(dist, 0.0, model.stds[j]);
            }
            let f: f64 = a.iter().zip(&model.w).map(|(&x, &w)| x * w).sum::<f64>() + model.b;
            let error: f64 = -(y[i] - f);

            for j in 0..model.k {
                model.w[j as usize] -= lr * a[j as usize] * error;
            }
            model.b -= lr * error;
        }
    }
}

pub fn tanh(x: f64) -> f64 {
    (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
}

pub fn predict(model: &mut RBFNet, X: &[Vec<f64>], is_classification: bool) -> Vec<f64> {
    let mut y_pred = Vec::new();
    for x in X {
        let mut a = vec![0.0; model.k as usize];
        for (j, c) in model.centers.iter().enumerate() {
            let dist = (0..x.len())
                .map(|k| (x[k] - c[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            a[j] = rbf(dist, 0.0, model.stds[j]);
        }
        let f: f64 = a.iter().zip(&model.w).map(|(&x, &w)| x * w).sum::<f64>() + model.b;
        if is_classification {
            y_pred.push(tanh(f));
        } else {
            y_pred.push(f);
        }
    }
    y_pred
}
