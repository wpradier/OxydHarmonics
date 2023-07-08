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

pub fn create_model(layers: &[usize]) -> RBFNet {
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

pub fn train_rbf(model: &mut RBFNet, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
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

pub fn predict_rbf(model: &mut RBFNet, X: &[Vec<f64>], is_classification: bool) -> Vec<f64> {
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

fn main() {
    let X_class = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
    let Y_class = vec![1.0, -1.0, -1.0];

    let X_reg = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let Y_reg = vec![2.0, 3.0, 4.0, 5.0];

    let layers = vec![2, 3, 1]; // Nombre de neurones
    let lr = 0.001; // Taux d'apprentissage
    let epochs = 500000; // Nombre d'itérations d'apprentissage

    let mut net = create_model(&layers);

    train_rbf(&mut net, &X_reg, &Y_reg, lr, epochs);

    for pred in &X_reg {
        let pred = predict_rbf(&mut net, &[pred.clone()], false)[0];
        println!("Prédiction : {:.2}", pred);
    }
}




/* use std::collections::HashMap;
use std::ops::Index;
use rand::Rng;
use tensorboard_rs::summary_writer::SummaryWriter;
use rand::distributions::uniform::Uniform;
use std::{f64::consts::E, usize};
use std::cmp::Ordering;

pub struct rbfModel {
    pub d: Vec<usize>,
    pub L: usize,
    pub W: Vec<Vec<Vec<f64>>>,
    pub X: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
    pub bias: f64,
    pub centroids: Vec<Vec<f64>>,
    pub variances: Vec<f64>,
}

pub fn create_rbf(npl: Vec<usize>, k: usize, features: &[Vec<f64>]) -> rbfModel {
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
                W[l][i].push(if j == 0 { random_number } else { random_number });
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

    let mut rng = rand::thread_rng();
    let dist = Uniform::new_inclusive(-1.0, 1.0);
    let bias = rng.sample(dist);

    let centroids = kmeans(features, k);
    let variances = compute_variances(&centroids, features);

    rbfModel {
        d: d,
        L: L,
        W: W,
        X: X,
        deltas: deltas,
        bias: bias,
        centroids: centroids,
        variances: variances,
    }
}

pub fn tanh(x: f64) -> f64 {
    (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
}

pub fn gaussian(x: f64) -> f64 {
    (-x.powf(2.0)).exp()
}



pub fn propagate_rbf<T>(model: &mut rbfModel, inputs: Vec<T>, is_classification: bool)
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
                if l < model.L {
                    let distance = euclidean_distance(&model.X[l - 1][1..], &model.centroids[j - 1]);

                    total = gaussian(distance / model.variances[j - 1]);
                } else if is_classification {
                    total = tanh(total);
                }
                model.X[l][j] = total;
            }
        }
    }

}

pub fn predict_rbf<T>(model: &mut rbfModel, input: T, is_classification: bool) -> Vec<f64>
where
    T: AsRef<[f64]>,
{
    let input_k = input.as_ref();
    propagate_rbf(model, vec![input_k], is_classification);
    model.X[model.L][1..].to_vec()
}

pub fn train_pmc<T, F>(
    mut model: &mut rbfModel,
    features: Vec<F>,
    labels: Vec<T>,
    is_classification: bool,
    num_iter: i32,
    alpha: f64,
) -> &mut rbfModel
where
    T: Index<usize, Output = f64> + AsRef<[f64]>,
    F: AsRef<[f64]> + Clone,
{
    let k = model.centroids.len();

    for _ in 0..num_iter {
        let mut rng = rand::thread_rng();
        let k_index = rng.gen_range(0..features.len());
        let input_k = &features[k_index];
        let y_k = &labels[k_index];

        propagate_rbf(&mut model, vec![input_k], is_classification);

        for j in 1..model.d[model.L] + 1 {
            model.deltas[model.L][j] = model.X[model.L][j] - y_k[j] as f64;

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

        // Mise à jour des centroids et variances
        let input_features: Vec<Vec<f64>> = features.iter().map(|input| input.as_ref().to_vec()).collect();
        model.centroids = kmeans(&input_features, k);
        model.variances = compute_variances(&model.centroids, &input_features);
    }

    model
}


    fn euclidean_distance(point1: &[f64], point2: &[f64]) -> f64 {
    let sum_squares = point1
    .iter()
    .zip(point2.iter())
    .map(|(&x, &y)| (x - y).powi(2))
    .sum::<f64>();
    sum_squares.sqrt()
    }
    
    fn compute_variances(centroids: &[Vec<f64>], points: &[Vec<f64>]) -> Vec<f64> {
    let num_clusters = centroids.len();
    let mut variances: Vec<f64> = vec![0.0; num_clusters];
    
    let mut cluster_points: HashMap<usize, Vec<&Vec<f64>>> = HashMap::new();

for point in points.iter() {
    let (closest_cluster, _) = centroids
        .iter()
        .enumerate()
        .min_by(|(_, c1), (_, c2)| {
            euclidean_distance(point, c1)
                .partial_cmp(&euclidean_distance(point, c2))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();

    cluster_points.entry(closest_cluster).or_default().push(point);
}

for (cluster, points) in cluster_points.iter() {
    let centroid = &centroids[*cluster];
    let variance_sum: f64 = points
        .iter()
        .map(|point| euclidean_distance(point, centroid).powi(2))
        .sum();
    let variance = variance_sum / points.len() as f64;
    variances[*cluster] = variance;
}

variances
}

fn kmeans(points: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
let num_points = points.len();
let num_dimensions = points[0].len();

let mut rng = rand::thread_rng();
let mut centroids: Vec<Vec<f64>> = Vec::new();

// Initialisation aléatoire des centroids
for _ in 0..k {
    let random_point = points[rng.gen_range(0..num_points)].clone();
    centroids.push(random_point);
}

// Itérations de l'algorithme K-means
loop {
    let mut cluster_points: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();

    // Attribution des points aux clusters les plus proches
    for point in points.iter() {
        let (closest_cluster, _) = centroids
            .iter()
            .enumerate()
            .min_by(|(_, c1), (_, c2)| {
                euclidean_distance(point, c1)
                    .partial_cmp(&euclidean_distance(point, c2))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();

        cluster_points.entry(closest_cluster).or_default().push(point.clone());
    }

    // Mise à jour des centroids
    let mut new_centroids: Vec<Vec<f64>> = Vec::new();
    let mut converged = true;

    for cluster in 0..k {
        let cluster_points = cluster_points.get(&cluster).unwrap();
        let num_cluster_points = cluster_points.len();

        if num_cluster_points == 0 {
            new_centroids.push(centroids[cluster].clone());
            continue;
        }

        let mut centroid: Vec<f64> = vec![0.0; num_dimensions];

        for point in cluster_points.iter() {
            for (i, &coord) in point.iter().enumerate() {
                centroid[i] += coord;
            }
        }

        for i in 0..num_dimensions {
            centroid[i] /= num_cluster_points as f64;
        }

        if !centroids[cluster].iter().zip(centroid.iter()).all(|(c1, c2)| {
            (c1 - c2).abs() < 0.0001
        }) {
            converged = false;
        }

        new_centroids.push(centroid);
    }

    if converged {
        break;
    }

    centroids = new_centroids;
}

centroids
}

    
        
     

fn main() {
    // Exemple de données d'entrée et d'étiquettes pour la classification
    let features = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let labels = vec![[0.0], [1.0], [1.0], [0.0]];

    // Configuration du modèle RBF
    let num_features = features[0].len();
    let num_hidden_units = 4;
    let num_iterations = 100000;
    let learning_rate = 0.01;

    // Création du modèle RBF
    let mut model = create_rbf(vec![num_features, num_hidden_units, 1],2,&features);

    // Entraînement du modèle
    train_pmc(
        &mut model,
        features.clone(),
        labels.clone(),
        true, // Classification
        num_iterations,
        learning_rate,
    );

    // Prédiction avec le modèle entraîné
    for input in &features {
        let prediction = predict_rbf(&mut model, input, true)[0];
        println!("Prédiction : {}", prediction);
    }
}
 */


/* use rand::prelude::{SliceRandom, ThreadRng};
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


pub fn create_model(k: i32, infer_stds: bool) -> RBFNet {
    let centers = Vec::new();
    let stds = Vec::new();
    let w = generate_random_weights(k as usize);
    let b = 0.0;

    RBFNet {
        k,
        infer_stds,
        centers,
        stds,
        w,
        b,
    }
}

fn generate_random_weights(k: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..k)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect::<Vec<f64>>()
}

pub fn train_rbf(model: &mut RBFNet, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
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

pub fn predict_rbf(model: &mut RBFNet, X: &[Vec<f64>], is_classification: bool) -> Vec<f64> {
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

fn main() {
    let X_class = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
    let Y_class = vec![1.0, -1.0, -1.0];

    let X_reg = vec![vec![1.0], vec![2.0],vec![3.0], vec![4.0]];
    let Y_reg = vec![2.0, 3.0, 4.0, 5.0];

    let k = 2; // Nombre de centres RBF
    let lr = 0.001; // Taux d'apprentissage
    let infer_stds = true; // Inférer les écarts-types
    let epochs = 100000; // Nombre d'itérations d'apprentissage

    let mut net = create_model(k, infer_stds);

    train_rbf(&mut net,&X_class, &Y_class, lr, epochs);

    for pred in &X_class {
        let pred = predict_rbf(&mut net,&[pred.clone()], false)[0];
        println!("Prédiction : {:.2}", pred);
    }
}



/*  */









use rand::prelude::{SliceRandom, thread_rng};
use rand::distributions::Uniform;
use std::f64::consts::PI;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Add;
use rand::Rng;

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

fn rbf(x: f64, c: f64, s: f64) -> f64 {
    (-1.0 / (2.0 * s.powi(2)) * (x - c).powi(2)).exp()
}

pub struct RBFNet {
    pub architecture: Vec<i32>,
    pub k: usize,
    pub infer_stds: bool,
    pub centers: Vec<Vec<f64>>,
    pub stds: Vec<f64>,
    pub w: Vec<Vec<Vec<f64>>>, 
    pub b: f64,
    pub layers: Vec<Vec<f64>>,
    pub X: Vec<Vec<f64>>,
}

impl RBFNet {
    pub fn create_model(architecture: Vec<i32>, infer_stds: bool) -> RBFNet {
        let k = architecture[architecture.len() - 1] as usize;
        let centers = Vec::new();
        let stds = Vec::new();
        let mut w = Vec::new();
        for i in 0..architecture.len() - 1 {
        let layer_size = architecture[i] as usize + 1;
        let next_layer_size = architecture[i + 1] as usize;
        let layer_weights = vec![vec![0.0; next_layer_size]; layer_size];
        w.push(layer_weights);
        }
        let b = 0.0;
        let layers = vec![vec![0.0; k]; architecture.len() - 1];
        let X = vec![vec![0.0; architecture[0] as usize]];

        RBFNet {
            architecture,
            infer_stds,
            centers,
            stds,
            w,
            b,
            layers,
            X,
            k,
        }
    }

    /* pub fn propagate_pmc(&mut self, inputs: Vec<Vec<f64>>, is_classification: bool) {
        let mut layers: Vec<Vec<f64>> = Vec::with_capacity(self.architecture.len());
        for _ in 0..self.architecture.len() {
            layers.push(vec![0.0; self.k]);
        }

        for input in inputs {
            let mut input_k = vec![0.0; self.architecture[0] as usize];
            let input_slice = input[1..].get(0..input_k.len()).and_then(|slice| Some(slice.to_owned())).unwrap_or_default();
            let mut input_k = vec![0.0; self.architecture[0] as usize];


            for l in 0..self.architecture.len() {
                for j in 1..self.architecture[l] + 1 {
                    let mut total = 0.0;
                    for i in 0..(self.architecture[l - 1] as usize + 1) {
                        total += self.w[l][i as usize][j as usize] * layers[l - 1][i as usize];
                    }
                    if l < self.architecture.len() - 1 || is_classification {
                        total = total.tanh();
                    }
                    layers[l][j as usize] = total;
                }
            }
            
        }
    }

    pub fn train_rbf(&mut self, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize, is_classification: bool) {
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
    
        // Calcul de la matrice 𝜙 et de 𝑊
        let mut phi = vec![vec![0.0; self.k]; n];
        for i in 0..n {
            for j in 0..self.k {
                let dist = (0..d)
                    .map(|k| (X[i][k] - self.centers[j][k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                phi[i][j] = rbf(dist, 0.0, self.stds[j]);
            }
        }
    
        let phi_transpose = transpose(&phi);
        let phi_transpose_phi = matrix_multiply(&phi_transpose, &phi);
        let phi_transpose_y = matrix_multiply_vector(&phi_transpose, &y);
        let inv_phi_transpose_phi = matrix_inverse(&phi_transpose_phi).expect("Impossible de calculer l'inverse de (𝜙𝑇𝜙)");
        self.w = vec![inv_phi_transpose_phi];
    
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
                let mut f: f64 = 0.0;
                for (a_i, w_i) in a.iter().zip(&self.w[0]) {
                    f += matrix_multiply_vector(&[vec![*a_i]], &w_i)[0];
                }
                f += self.b;




                let error: f64 = match is_classification {
                    true => sigmoid(f) - y[i],
                    false => y[i] - f,
                };
    
                for j in 0..(self.k - 1) {
                    for i in 0..(self.architecture.len() - 1) {
                        for k in 0..(self.architecture[i] as usize + 1) {
                            if i == 0 {
                                self.w[i][j][k] += lr * a[j] * error;
                            } else {
                                let derivative = if i < (self.architecture.len() - 2) || is_classification {
                                    1.0 - self.layers[i][j].powi(2)
                                } else {
                                    1.0
                                };
                                self.w[i][j][k] += lr * a[j] * error * derivative * self.layers[i - 1][k];
                            }
                        }
                    }
                }
                self.b += lr * error;
            }
        }
    } */

    pub fn propage_rbf(&mut self, inputs: &[Vec<f64>]) {
        let n = inputs.len();
        let mut layer_inputs = inputs.to_vec();

        for i in 0..self.architecture.len() - 1 {
            let layer_size = self.architecture[i] as usize + 1;
            let next_layer_size = self.architecture[i + 1] as usize;

            let mut layer_outputs = vec![vec![0.0; next_layer_size]; n];

            for j in 0..n {
                for k in 0..next_layer_size {
                    let rbf_outputs = layer_inputs[j]
                        .iter()
                        .zip(&self.centers[i])
                        .zip(&self.stds[i])
                        .map(|((&x, &c), &s)| rbf(x, c, s))
                        .collect::<Vec<f64>>();

                    let weighted_sum = rbf_outputs.iter().zip(&self.w[i][..][k][..]).fold(0.0, |acc, (&o, &weight)| {
                        acc + o * weight
                    });

                    layer_outputs[j][k] = sigmoid(weighted_sum + self.b);
                }
            }

            if i < self.architecture.len() - 2 {
                let bias_column = vec![1.0; n];
                layer_inputs = [bias_column, layer_outputs].concat();
            } else {
                self.layers = layer_outputs;
            }
        }
    }

    pub fn train_rbf(&mut self, X: &[Vec<f64>], y: &[f64], learning_rate: f64, num_epochs: usize, verbose: bool) {
        let n = X.len();

        // K-means clustering
        let (centers, stds) = kmeans(X, self.k);
        self.centers = centers;
        self.stds = stds;

        // Initialize weights randomly
        let mut rng = thread_rng();
        for i in 0..self.architecture.len() - 1 {
            let layer_size = self.architecture[i] as usize + 1;
            let next_layer_size = self.architecture[i + 1] as usize;

            for j in 0..layer_size {
                for k in 0..next_layer_size {
                    self.w[i][j][k] = rng.gen_range(-0.5..0.5);
                }
            }
        }
        fn sigmoid_derivative(x: f64) -> f64 {
            let sigmoid_x = sigmoid(x);
            sigmoid_x * (1.0 - sigmoid_x)
        }
        
        // Training loop
        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for i in 0..n {
                self.propage_rbf(&[X[i].clone()]);

                let prediction = self.layers[self.layers.len() - 1][0];
                let error = y[i] - prediction;

                epoch_loss += error.powi(2);

                let delta = learning_rate * error * sigmoid_derivative(prediction);

                for j in (0..self.architecture.len() - 1).rev() {
                    let layer_inputs = if j > 0 {
                        [vec![1.0], self.layers[j - 1].clone()].concat()
                    } else {
                        X[i].clone()
                    };

                    let layer_outputs = self.layers[j].clone();

                    for k in 0..layer_inputs.len() {
                        for l in 0..layer_outputs.len() {
                            let delta_w = delta * layer_inputs[k] * sigmoid_derivative(layer_outputs[l]);
                            self.w[j][k][l] += delta_w;
                        }
                    }

                    if j > 0 {
                        let mut next_delta = vec![0.0; layer_inputs.len()];
                        for k in 0..layer_inputs.len() {
                            let weights = self.w[j][k].clone();
                            let weighted_deltas = weights.iter().zip(&next_delta).map(|(&weight, &delta)| weight * delta).collect::<Vec<f64>>();
                            next_delta[k] = sigmoid_derivative(layer_inputs[k]) * weighted_deltas.iter().sum::<f64>();
                        }
                        delta = next_delta;
                    }
                }
            }

            let mean_loss = epoch_loss / n as f64;

            if verbose && (epoch + 1) % 100 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch + 1, mean_loss);
            }
        }
    }

    pub fn predict(&mut self, input: &[f64]) -> f64 {
        self.propagate_pmc(vec![input.to_vec()], false);
        self.layers[self.layers.len() - 1][1]
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

fn matrix_multiply(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = matrix1.len();
    let n = matrix2[0].len();
    let p = matrix1[0].len();
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    result
}

fn matrix_multiply_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut result = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    result
}



fn matrix_inverse(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut augmented = matrix.to_vec();
    for i in 0..n {
        for j in n..2 * n {
            augmented[i].push(if i == j - n { 1.0 } else { 0.0 });
        }
    }
    for i in 0..n {
        let pivot = augmented[i][i];
        if pivot.abs() < 1e-10 {
            return None; // Matrix is not invertible
        }
        for j in i..2 * n {
            augmented[i][j] /= pivot;
        }
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in i..2 * n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in n..2 * n {
            inverse[i][j - n] = augmented[i][j];
        }
    }
    Some(inverse)
}


fn main() {
    // Exemple d'utilisation pour la classification XOR
    let architecture = vec![2, 3, 1];
    let mut net = RBFNet::create_model(architecture.clone(), true);
    let X = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];
    net.train_rbf(&X, &y, 0.1, 1000, true);

    // Test de classification XOR
    println!("Classification XOR:");
    for i in 0..X.len() {
        let input = &X[i];
        let output = net.predict(input);
        println!("Input: {:?} => Output: {:.3}", input, output);
    }

    // Exemple d'utilisation pour la régression sin(x)
    let architecture = vec![1, 10, 1];
    let mut net = RBFNet::create_model(architecture.clone(), true);
    let X: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![(i as f64 / 50.0 - 1.0) * PI])
        .collect();
    let y: Vec<f64> = X.iter().map(|x| (x[0]).sin()).collect();
    net.train_rbf(&X, &y, 0.1, 1000,false);

    // Test de régression sin(x)
    println!("\nRégression sin(x):");
    for i in 0..X.len() {
        let input = &X[i];
        let output = net.predict(input);
        println!("Input: {:.3} => Output: {:.3}", input[0], output);
    }
}

  */
 



 







/* use rand::prelude::{SliceRandom, thread_rng};
use rand::distributions::Uniform;
use std::f64::consts::PI;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Add;

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

fn rbf(x: f64, c: f64, s: f64) -> f64 {
    (-1.0 / (2.0 * s.powi(2)) * (x - c).powi(2)).exp()
}

pub struct RBFNet {
    pub architecture: Vec<i32>,
    pub k: usize,
    pub infer_stds: bool,
    pub centers: Vec<Vec<f64>>,
    pub stds: Vec<f64>,
    pub w: Vec<Vec<Vec<f64>>>, 
    pub b: f64,
    pub layers: Vec<Vec<f64>>,
    pub X: Vec<Vec<f64>>,
}

impl RBFNet {
    pub fn create_model(architecture: Vec<i32>, infer_stds: bool) -> RBFNet {
        let k = architecture[architecture.len() - 1] as usize;
        let centers = Vec::new();
        let stds = Vec::new();
        let mut w = Vec::new();
        for i in 0..architecture.len() - 1 {
        let layer_size = architecture[i] as usize + 1;
        let next_layer_size = architecture[i + 1] as usize;
        let layer_weights = vec![vec![0.0; next_layer_size]; layer_size];
        w.push(layer_weights);
        }
        let b = 0.0;
        let layers = vec![vec![0.0; k]; architecture.len() - 1];
        let X = vec![vec![0.0; architecture[0] as usize]];

        RBFNet {
            architecture,
            infer_stds,
            centers,
            stds,
            w,
            b,
            layers,
            X,
            k,
        }
    }

    pub fn propagate_pmc(&mut self, inputs: Vec<Vec<f64>>, is_classification: bool) {
        let mut layers: Vec<Vec<f64>> = Vec::with_capacity(self.architecture.len());
        for _ in 0..self.architecture.len() {
            layers.push(vec![0.0; self.k]);
        }

        for input in inputs {
            let mut input_k = vec![0.0; self.architecture[0] as usize];
            input_k.copy_from_slice(&input[1..]);
            layers[0][1..].copy_from_slice(&input_k);

            for l in 1..self.architecture.len() {
                for j in 1..self.architecture[l] + 1 {
                    let mut total = 0.0;
                    for i in 0..self.architecture[l - 1] + 1 {
                        total += self.w[l][i as usize][j as usize] * layers[l - 1][i as usize];
                    }
                    if l < self.architecture.len() - 1 || is_classification {
                        total = total.tanh();
                    }
                    layers[l][j as usize] = total;
                }
            }
        }
    }

    pub fn train_rbf(&mut self, X: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize, is_classification: bool) {
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
    
        // Calcul de la matrice 𝜙 et de 𝑊
        let mut phi = vec![vec![0.0; self.k]; n];
        for i in 0..n {
            for j in 0..self.k {
                let dist = (0..d)
                    .map(|k| (X[i][k] - self.centers[j][k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                phi[i][j] = rbf(dist, 0.0, self.stds[j]);
            }
        }
    
        let phi_transpose = transpose(&phi);
        let phi_transpose_phi = matrix_multiply(&phi_transpose, &phi);
        let phi_transpose_y = matrix_multiply_vector(&phi_transpose, &y);
        let inv_phi_transpose_phi = matrix_inverse(&phi_transpose_phi).expect("Impossible de calculer l'inverse de (𝜙𝑇𝜙)");
        self.w = vec![inv_phi_transpose_phi];
    
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
                let mut f: f64 = 0.0;
                for (a_i, w_i) in a.iter().zip(&self.w[0]) {
                    f += matrix_multiply_vector(&[vec![*a_i]], &w_i)[0];
                }
                f += self.b;




                let error: f64 = match is_classification {
                    true => sigmoid(f) - y[i],
                    false => y[i] - f,
                };
    
                for j in 0..self.k {
                    for i in 0..self.architecture.len() - 1 {
                        for k in 0..self.architecture[i] as usize + 1 {
                            if i == 0 {
                                self.w[i][j][k] += lr * a[j] * error;
                            } else {
                                let derivative = if i < self.architecture.len() - 2 || is_classification {
                                    1.0 - self.layers[i][j].powi(2)
                                } else {
                                    1.0
                                };
                                self.w[i][j][k] += lr * a[j] * error * derivative * self.layers[i - 1][k];
                            }
                        }
                    }
                }
                self.b += lr * error;
            }
        }
    }

    pub fn predict(&mut self, input: &[f64]) -> f64 {
        self.propagate_pmc(vec![input.to_vec()], false);
        self.layers[self.layers.len() - 1][1]
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

fn matrix_multiply(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = matrix1.len();
    let n = matrix2[0].len();
    let p = matrix1[0].len();
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    result
}

fn matrix_multiply_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut result = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    result
}



fn matrix_inverse(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut augmented = matrix.to_vec();
    for i in 0..n {
        for j in n..2 * n {
            augmented[i].push(if i == j - n { 1.0 } else { 0.0 });
        }
    }
    for i in 0..n {
        let pivot = augmented[i][i];
        if pivot.abs() < 1e-10 {
            return None; // Matrix is not invertible
        }
        for j in i..2 * n {
            augmented[i][j] /= pivot;
        }
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in i..2 * n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in n..2 * n {
            inverse[i][j - n] = augmented[i][j];
        }
    }
    Some(inverse)
}


fn main() {
    // Exemple d'utilisation pour la classification XOR
    let architecture = vec![2, 3, 1];
    let mut net = RBFNet::create_model(architecture.clone(), true);
    let X = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];
    net.train_rbf(&X, &y, 0.1, 1000, true);

    // Test de classification XOR
    println!("Classification XOR:");
    for i in 0..X.len() {
        let input = &X[i];
        let output = net.predict(input);
        println!("Input: {:?} => Output: {:.3}", input, output);
    }

    // Exemple d'utilisation pour la régression sin(x)
    let architecture = vec![1, 10, 1];
    let mut net = RBFNet::create_model(architecture.clone(), true);
    let X: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![(i as f64 / 50.0 - 1.0) * PI])
        .collect();
    let y: Vec<f64> = X.iter().map(|x| (x[0]).sin()).collect();
    net.train_rbf(&X, &y, 0.1, 1000,false);

    // Test de régression sin(x)
    println!("\nRégression sin(x):");
    for i in 0..X.len() {
        let input = &X[i];
        let output = net.predict(input);
        println!("Input: {:.3} => Output: {:.3}", input[0], output);
    }
}


 */






/* use rand::prelude::{SliceRandom, ThreadRng};
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
            for (j, c) in self.centers.iter().enumerate() {
                let dist = (0..x.len())
                    .map(|k| (x[k] - c[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                a[j] = rbf(dist, 0.0, self.stds[j]);
            }
           
                let f: f64 = a.iter().zip(&self.w).map(|(&x, &w)| x * w).sum::<f64>() + self.b;
                if is_classification {
                    y_pred.push(f.signum());
                } else {
                    y_pred.push(f);
                }
            }
            y_pred
        }
        
    }
 */
//====================================================================================
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


