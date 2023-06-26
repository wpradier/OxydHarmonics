use rand::prelude::SliceRandom;
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
    let mut rng = rand::thread_rng();
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
            let mut min_dist = f64::MAX;
            let mut min_index = 0;
            for i in 0..k {
                let dist = (0..d)
                    .map(|j| (x[j] - centers[i][j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_index = i;
                }
            }
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
        let mut min_dist = f64::MAX;
        for i in 0..k {
            let dist = (0..d)
                .map(|j| (x[j] - centers[i][j]).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
        std_sum += min_dist;
        count += 1;
    }
    let mean_std = std_sum / count as f64;
    for i in 0..k {
        stds[i] = mean_std;
    }

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
        self.centers = centers;
        self.stds = stds;

        self.w = vec![0.0; self.k];
        self.b = 0.0;

        for _ in 0..epochs {
            for i in 0..n {
                let mut a = vec![0.0; self.k];
                for (j, c) in self.centers.iter().enumerate() {
                    let dist = (0..d)
                        .map(|k| (X[i][k] - c[k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    a[j] = rbf(dist, 0.0, self.stds[j]);
                }
                let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
                let loss: f64 = (y[i] - f).powi(2);
                println!("Loss: {:.2}", loss);

                let error: f64 = -(y[i] - f);
                for j in 0..self.k {
                    self.w[j] -= lr * a[j] * error;
                }
                self.b -= lr * error;
            }
        }
    }

    fn predict(&self, X: &[Vec<f64>]) -> Vec<f64> {
        let mut y_pred = Vec::new();
        for x in X {
            let mut a = vec![0.0; self.k];
            for (j, c) in self.centers.iter().enumerate() {
                let dist = (0..x.len()).map(|k| (x[k] - c[k]).powi(2)).sum::<f64>().sqrt();
                a[j] = rbf(dist, 0.0, self.stds[j]);
            }
            let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
            y_pred.push(f);
        }
        y_pred
    }
}

fn main() {
    let X = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
    ];

    let Y = vec![
        1.0,
        -1.0,
        -1.0,
    ];

    let k = 2; // Nombre de centres RBF
    let lr = 0.01; // Taux d'apprentissage
    let infer_stds = true; // Inférer les écarts-types
    let epochs = 100_000; // Nombre d'itérations d'apprentissage

    let mut net = RBFNet::create_model(k, infer_stds);

    net.train(&X, &Y, lr, epochs);

    let X_test = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
    ];

    let y_pred = net.predict(&X_test);

    for (i, pred) in y_pred.iter().enumerate() {
        println!("Prédiction {} : {:.2}", i + 1, pred);
    }
}

