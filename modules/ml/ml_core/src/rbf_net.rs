use rand::prelude::{SliceRandom, ThreadRng};
use rand::thread_rng;
use rand::Rng;
use std::f64::consts::PI;

fn rbf(x: f64, c: f64, s: f64) -> f64 {
    (-1.0 / (2.0 * s.powi(2)) * (x - c).powi(2)).exp()
}

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

struct RBFNet {
    k: usize,
    infer_stds: bool,
    centers: Vec<Vec<f64>>,
    stds: Vec<f64>,
    w: Vec<f64>,
    b: f64,
}

impl RBFNet {
    fn create_model(k: usize, infer_stds: bool) -> Self {
        RBFNet {
            k,
            infer_stds,
            centers: Vec::new(),
            stds: Vec::new(),
            w: Vec::new(),
            b: 0.0,
        }
    }

    fn train(&mut self, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
        let n = X.len();
        let d = X[0].len();
        let (centers, stds) = if self.infer_stds {
            kmeans(X, self.k)
        } else {
            (self.centers.clone(), self.stds.clone())
        };

        // Vérification de la taille des vecteurs dans centers
        for center in &centers {
            if center.len() != d {
                panic!("Les vecteurs dans centers doivent avoir la même taille que les vecteurs dans X.");
            }
        }

        self.centers = centers;
        self.stds = stds;

        self.w = vec![0.0; self.k];
        self.b = 0.0;

        let mut rng = thread_rng();
        for _ in 0..epochs {
            let random_indices: Vec<usize> = (0..n).collect();
            let shuffled_indices: Vec<usize> = random_indices.choose_multiple(&mut rng, n).cloned().collect();
            for &i in &shuffled_indices {
                let mut a = vec![0.0; self.k];
                for (j, c) in self.centers.iter().enumerate() {
                    let dist = (0..d)
                        .map(|k| (X[i][k] - c[k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    a[j] = rbf(dist, 0.0, self.stds[j]);
                }
                let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
                let error: f64 = -(y[i] - f);

                for j in 0..self.k {
                    self.w[j] -= lr * a[j] * error;
                }
                self.b -= lr * error;
            }
        }
    }

    fn predict(&self, X: &[Vec<f64>], is_classification: bool) -> Vec<f64> {
        let mut y_pred = Vec::new();
        for x in X {
            let mut a = vec![0.0; self.k];
            let mut valid = true; // Indicateur de validité des vecteurs
            for (j, c) in self.centers.iter().enumerate() {
                let dist = (0..x.len())
                    .map(|k| (x[k] - c[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                a[j] = rbf(dist, 0.0, self.stds[j]);
            }
            if valid {
                let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
                if is_classification {
                    y_pred.push(f.signum());
                } else {
                    y_pred.push(f);
                }
            }
        }
        y_pred
    }
}
 fn main() {
    let X_class = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
    let Y_class = vec![1.0, -1.0, -1.0];

    let X_reg = vec![vec![1.0], vec![2.0]];
    let Y_reg = vec![2.0, 3.0]; // Modify Y_reg to be a slice of Vec<f64>

    let k = 2; // Nombre de centres RBF
    let lr = 0.01; // Taux d'apprentissage
    let infer_stds = true; // Inférer les écarts-types
    let epochs = 100000; // Nombre d'itérations d'apprentissage

    let mut net = RBFNet::create_model(k, infer_stds);

    net.train(&X_class, &Y_class, lr, epochs);

    for pred in &X_class {
        let pred = net.predict(&[pred.clone()], false)[0];
        println!("Prédiction : {:.2}", pred);
    }
} 

/* use std::collections::HashMap;
use std::ops::Index;
use rand::Rng;
use rand::distributions::uniform::Uniform;
use std::f64::consts::E;

pub struct RbfModel {
    d: Vec<usize>,
    L: usize,
    W: Vec<Vec<f64>>,
    X: Vec<Vec<f64>>,
    deltas: Vec<f64>,
}

impl RbfModel {
    pub fn new(num_centers: usize, num_features: usize) -> RbfModel {
        let d = vec![num_features, num_centers];
        let L = 1;
        let mut W: Vec<Vec<f64>> = Vec::new();
        let mut rng = rand::thread_rng();
        let dist = Uniform::new_inclusive(-1.0, 1.0);

        for l in 0..L + 1 {
            W.push((0..num_centers)
                .map(|_| rng.sample(dist))
                .collect::<Vec<f64>>());
        }

        let X = vec![Vec::new(); L + 1];
        let deltas = vec![0.0; num_centers];

        RbfModel {
            d,
            L,
            W,
            X,
            deltas,
        }
    }

    pub fn gaussian_kernel(&self, x: &[f64], center: &[f64], sigma: f64) -> f64 {
        let distance_sq = x.iter()
            .zip(center.iter())
            .map(|(&xi, &ci)| (xi - ci).powi(2))
            .sum::<f64>();

        (-distance_sq / (2.0 * sigma.powi(2))).exp()
    }

    pub fn tanh(&self, x: f64) -> f64 {
        (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
    }

    pub fn propagate(&mut self, inputs: &[f64], is_classification: bool) {
        let num_centers = self.d[1];
        self.X[1] = vec![0.0; num_centers];

        for j in 0..num_centers {
            self.X[1][j] = self.gaussian_kernel(inputs, &self.W[0], 1.0);
            if is_classification {
                self.X[1][j] = self.tanh(self.X[1][j]);
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> f64 {
        self.propagate(inputs, is_classification);
        self.X[self.L][0]
    }

    pub fn train<F>(
        &mut self,
        features: &[F],
        labels: &[f64],
        num_iter: i32,
        alpha: f64,
        is_classification: bool,
    ) where
        F: AsRef<[f64]>,
    {
        let mut map = HashMap::new();

        for _ in 0..num_iter {
            let k = rand::thread_rng().gen_range(0..features.len());
            let input_k = &features[k];
            let y_k = labels[k];

            self.propagate(input_k.as_ref(), is_classification);

            let delta = self.X[self.L][0] - y_k;
            self.deltas[0] = delta;

            for i in 0..self.d[0] {
                let xi = input_k.as_ref()[i];
                let delta_w = alpha * delta * xi;
                self.W[0][i] -= delta_w;
            }
        }

        let predictions = features
            .iter()
            .map(|input| self.predict(input.as_ref(), is_classification))
            .collect::<Vec<f64>>();

        let loss = self.calculate_loss(&predictions, labels, is_classification);
        map.insert("loss".to_string(), loss as f32);
        //writer.add_scalars(&log_path, &map, num_iter as usize);
    }

    pub fn calculate_loss(&self, predictions: &[f64], labels: &[f64], is_classification: bool) -> f64 {
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
}

fn main() {
    let X_class = vec![[1.0, 1.0], [2.0, 3.0], [3.0, 3.0]];
    let Y_class: Vec<f64> = vec![1.0, -1.0, -1.0];

    let X_reg = vec![vec![1.0], vec![2.0]];
    let Y_reg: Vec<f64> = vec![2.0, 3.0];

    let k = 3; // Nombre de centres RBF
    let lr = 0.001; // Taux d'apprentissage
    let infer_stds = true; // Inférer les écarts-types
    let epochs = 100000; // Nombre d'itérations d'apprentissage

    // Entraînement pour la régression
    let mut rbf_reg = RbfModel::new(k, X_reg[0].len());
    rbf_reg.train(&X_reg, &Y_reg, epochs, lr, false);

    println!("Prédictions pour la régression :");
    for pred in &X_reg {
        let pred_value = rbf_reg.predict(pred, false);
        println!("Prédiction : {:.2}", pred_value);
    }

    // Entraînement pour la classification binaire
    let mut rbf_class = RbfModel::new(k, X_class[0].len());
    rbf_class.train(&X_class, &Y_class, epochs, lr, true);

    println!("Prédictions pour la classification :");
    for pred in &X_class {
        let pred_value = rbf_class.predict(pred, true);
        let class_label = if pred_value > 0.0 { 1 } else { -1 };
        println!("Prédiction : {}", class_label);
    }
} */


