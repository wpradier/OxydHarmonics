use rand::Rng;
use std::f64::consts::PI;
use rand::prelude::SliceRandom;

fn rbf(x: f64, c: f64, s: f64) -> f64 {
    (-1.0 / (2.0 * s.powi(2)) * (x - c).powi(2)).exp()
}

fn kmeans(X: &[Vec<f64>], k: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = X.len();
    let mut centers = vec![vec![0.0; X[0].len()]; k];
    let mut stds = vec![0.0; k];

    // Initialisation aléatoire des centres
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    for i in 0..k {
        centers[i] = X[indices[i]].clone();
    }

    // Boucle principale du k-means
    let mut prev_centers = vec![vec![0.0; X[0].len()]; k];
    while centers != prev_centers {
        prev_centers.clone_from_slice(&centers);
        let mut counts = vec![0; k];
        let mut sums = vec![vec![0.0; X[0].len()]; k];
        for x in X {
            let mut min_dist = f64::MAX;
            let mut min_index = 0;
            for i in 0..k {
                let dist = (x.iter().zip(&centers[i]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>()).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_index = i;
                }
            }
            for (j, &val) in x.iter().enumerate() {
                sums[min_index][j] += val;
            }
            counts[min_index] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..X[0].len() {
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
            let dist = (x.iter().zip(&centers[i]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>()).sqrt();
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
    fn new(k: usize, infer_stds: bool) -> Self {
        RBFNet {
            k,
            infer_stds,
            centers: Vec::new(),
            stds: Vec::new(),
            w: Vec::new(),
            b: 0.0,
        }
    }

    fn fit(&mut self, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
        let n = X.len();
        let (centers, stds) = if self.infer_stds {
            kmeans(X, self.k)
        } else {
            kmeans(X, self.k)
        };
        self.centers = centers;
        self.stds = stds;
    
        self.w = vec![0.0; self.k];
        self.b = 0.0;
    
        for _ in 0..epochs {
            for i in 0..n {
                for (c, &s) in self.centers.iter().zip(self.stds.iter()) {
                    let a: Vec<f64> = X[i].iter().zip(c.iter()).map(|(&x, &ci)| rbf(x, ci, s)).collect();
                    let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
                    let loss: f64 = (y[i] - f).powi(2);
                    println!("Loss: {:.2}", loss);
    
                    let error: f64 = -(y[i] - f);
                    self.w = self
                        .w
                        .iter()
                        .zip(&a)
                        .map(|(&w, &x)| w - lr * x * error)
                        .collect();
                    self.b -= lr * error;
                }
            }
        }
    }
    

    fn predict(&self, X: &[Vec<f64>]) -> Vec<f64> {
        let mut y_pred = Vec::new();
        for x in X {
            let a: Vec<f64> = self
                .w
                .iter()
                .zip(self.centers.iter())
                .zip(self.stds.iter())
                .map(|((&w, c), &s)| rbf(x[0], c[0], s) * w)
                .collect();
            let f: f64 = a.iter().sum::<f64>() + self.b;
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

    let mut net = RBFNet::new(k, infer_stds);

    net.fit(&X, &Y, lr, epochs);

    let X_test = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
    ];

    let y_pred = net.predict(&X_test);

    for (i, pred) in y_pred.iter().enumerate() {
        println!("Prédiction {} : {:.2}", i + 1, pred);
    }
}

    







/* use std::collections::HashMap;
extern crate rusty_machine as rm;
extern crate rulinalg;

use rm::learning::gp::{GaussianProcess};
use rm::learning::SupModel;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

struct KMeans {
    k: usize,
    tol: f64,
    max_iter: usize,
    centroids: HashMap<usize, Vec<f64>>,
    classifications: HashMap<usize, Vec<Vec<f64>>>,
}

impl KMeans {
    fn new(k: usize, tol: f64, max_iter: usize) -> Self {
        KMeans {
            k, //Le nombre de clusters que nous souhaitons créer.
            tol, //La tolérance, c'est-à-dire la différence maximale entre les coordonnées d'un centroïde à deux itérations successives.
            max_iter,
            centroids: HashMap::new(), //Une structure de données HashMap qui associe à chaque identifiant de cluster un vecteur représentant les coordonnées du centroïde correspondant.
            classifications: HashMap::new(), //Une structure de données HashMap qui associe à chaque identifiant de cluster un vecteur de vecteurs représentant les coordonnées des points de données qui lui sont associés.
        }
    }

    fn train(&mut self, data: &[Vec<f64>]) {
        for i in 0..self.k {
            self.centroids.insert(i, data[i].clone());
        }

        for _ in 0..self.max_iter {
            self.classifications.clear();

            for i in 0..self.k {
                self.classifications.insert(i, Vec::new());
            }

            for featureset in data {
                let distances: Vec<f64> = self
                    .centroids
                    .iter()
                    .map(|(_, centroid)| euclidean_distance(featureset, centroid))
                    .collect();
                let classification = distances
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.classifications.get_mut(&classification).unwrap().push(featureset.clone());
            }

            let prev_centroids = self.centroids.clone();

            for (classification, featuresets) in self.classifications.iter_mut() {
                let centroid = calculate_centroid(featuresets);
                self.centroids.insert(*classification, centroid);
            }

            let mut optimized = true;

            for (classification, current_centroid) in self.centroids.iter() {
                let original_centroid = prev_centroids.get(classification).unwrap();
                let diff = vector_difference(original_centroid, current_centroid);
                let percent_change = calculate_percent_change(diff, original_centroid);
                if percent_change > self.tol {
                    println!("{}", percent_change);
                    optimized = false;
                }
            }

            if optimized {
                break;
            }
        }
    }

    fn predict(&self, data: &[f64]) -> usize {
        let distances: Vec<f64> = self
            .centroids
            .iter()
            .map(|(_, centroid)| euclidean_distance(data, centroid))
            .collect();
        let classification = distances
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        classification
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn calculate_centroid(featuresets: &[Vec<f64>]) -> Vec<f64> {
    let num_features = featuresets[0].len();
    let num_samples = featuresets.len();
    let mut centroid = vec![0.0; num_features];

    for featureset in featuresets {
        for (i, &value) in featureset.iter().enumerate() {
            centroid[i] += value;
        }
    }

    for value in &mut centroid {
        *value /= num_samples as f64;
    }

    centroid
}

fn vector_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

fn calculate_percent_change(diff: Vec<f64>, original: &[f64]) -> f64 {
    diff.iter()
        .zip(original.iter())
        .map(|(&x, &y)| x / y * 100.0)
        .sum()
}
 */
/* fn main() {
    let data = vec![
        vec![1.0, 2.0],
        vec![5.0, 8.0],
        vec![1.5, 1.8],
        vec![8.0, 8.0],
        vec![1.0, 0.6],
        vec![9.0, 11.0],
    ];

    let mut model = KMeans::new(2, 0.001, 300);
    model.fit(&data);

    for (classification, centroid) in model.centroids.iter() {
        println!(
            "Centroid {}: ({}, {})",
            classification, centroid[0], centroid[1]
        );
    }

    for (classification, featuresets) in model.classifications.iter() {
        for featureset in featuresets {
            println!(
                "Classification {}: ({}, {})",
                classification, featureset[0], featureset[1]
            );
        }
    }
} */

/* 


fn main() {
    // Données d'entraînement pour KMeans
    let data = vec![
        vec![1.0, 2.0],
        vec![5.0, 8.0],
        vec![1.5, 1.8],
        vec![8.0, 8.0],
        vec![1.0, 0.6],
        vec![9.0, 11.0],
    ];

    let mut kmeans_model = KMeans::new(2, 0.001, 300);
    kmeans_model.train(&data);

    for (classification, centroid) in kmeans_model.centroids.iter() {
        println!(
            "Centroid {}: ({}, {})",
            classification, centroid[0], centroid[1]
        );
    }

    for (classification, featuresets) in kmeans_model.classifications.iter() {
        for featureset in featuresets {
            println!(
                "Classification {}: ({}, {})",
                classification, featureset[0], featureset[1]
            );
        }
    }

    // Données d'entraînement pour GaussianProcess
    let inputs = rm::prelude::Matrix::new(6, 2, vec![
        1.0, 1.0,
        2.0, 2.0,
        3.0, 2.5,
         8.0, 7.0,
        9.0, 9.0,
        7.0, 9.5,
    ]);

    let targets = rm::prelude::Vector::new(vec![
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]);

    // Création du modèle GaussianProcess
    let mut gp_model = GaussianProcess::default();

    // Entraînement du modèle avec les données d'entraînement
    gp_model.train(&inputs, &targets).unwrap();

    // Exemple d'utilisation du modèle GaussianProcess
    let test_input = rm::prelude::Matrix::new(1, 2, vec![4.0, 5.0]); // Valeur de test
    let prediction = gp_model.predict(&test_input); // Prédiction du modèle

    println!("Prediction: {:?}", prediction);
}


 */