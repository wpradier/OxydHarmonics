extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::Rng;

fn kmeans<X>(x: Array2<X>, k: usize) -> (Array1<X>, Array1<f64>)
where
    X: std::clone::Clone + std::cmp::PartialOrd + std::ops::AddAssign<X> + std::ops::Div<X, Output = X> + std::ops::Sub<X, Output = X> + ndarray::ScalarOperand,
{
    let x_squeezed = x.into_dimensionality::<Ix1>().unwrap();
    let mut rng = rand::thread_rng();
    let clusters = Array::from_shape_fn(k, |_| x_squeezed[rng.gen_range(0, x_squeezed.len())].clone());
    let mut prev_clusters = clusters.clone();
    let mut stds = Array::zeros(k);
    let mut converged = false;

    while !converged {
        let distances = x_squeezed
            .into_shape((x_squeezed.len(), 1))
            .unwrap()
            .broadcast((x_squeezed.len(), k))
            .unwrap()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .unwrap()
            .reversed_axes();

        distances.zip_mut_with(&clusters, |dist, &cluster| {
            *dist = (dist.clone() - cluster).abs();
        });

        let closest_cluster = distances
            .argmin_axis(Axis(1))
            .into_shape(x_squeezed.len())
            .unwrap();

        for i in 0..k {
            let points_for_cluster = x_squeezed.to_owned().into_iter()
                .zip(closest_cluster.iter().cloned())
                .filter(|&(_, cluster)| cluster == i)
                .map(|(point, _)| point)
                .collect::<Vec<_>>();

            if points_for_cluster.len() > 0 {
                clusters[i] = points_for_cluster.iter().sum::<X>() / X::from(points_for_cluster.len()).unwrap();
            }
        }

        converged = (&clusters - &prev_clusters).norm() < 1e-6;
        prev_clusters = clusters.clone();
    }

    let distances = x_squeezed
        .into_shape((x_squeezed.len(), 1))
        .unwrap()
        .broadcast((x_squeezed.len(), k))
        .unwrap()
        .to_owned()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .reversed_axes();

    distances.zip_mut_with(&clusters, |dist, &cluster| {
        *dist = (dist.clone() - cluster).abs();
    });

    let closest_cluster = distances
        .argmin_axis(Axis(1))
        .into_shape(x_squeezed.len())
        .unwrap();

    let mut clusters_with_no_points = Vec::new();

    for i in 0..k {
        let points_for_cluster = x_squeezed.to_owned().into_iter()
            .zip(closest_cluster.iter().cloned())
            .filter(|&(_, cluster)| cluster == i)
            .map(|(point, _)| point)
            .collect::<Vec<_>>();

        if points_for_cluster.len() < 2 {
            clusters_with_no_points.push(i);
            continue;
        } else {
            stds[i] = points_for_cluster.iter()
                .map(|&point| (point - clusters[i]).to_owned())
                .collect::<Array1<_>>()
                .std_dev();
        }
    }

    if clusters_with_no_points.len() > 0 {
        let points_to_average = (0..k)
            .filter(|&i| !clusters_with_no_points.contains(&i))
            .flat_map(|i| {
                x_squeezed.to_owned().into_iter()
                    .zip(closest_cluster.iter().cloned())
                    .filter(|&(_, cluster)| cluster == i)
                    .map(|(point, _)| point)
                    .collect::<Vec<_>>()
            })
            .collect::<Array1<_>>();

        stds[clusters_with_no_points] = points_to_average.std_dev();
    }

    (clusters, stds)
}

fn main() {
    let x: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0], [11.0], [12.0], [13.0], [14.0], [15.0]];
    let k = 2;
    let (clusters, stds) = kmeans(x, k);
    println!("Clusters: {:?}", clusters);
    println!("Standard Deviations: {:?}", stds);
}
