use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray::{Array, array, Array1, Array2, ArrayBase, Axis, concatenate, Ix2, s, ArrayView, OwnedRepr};

pub struct LinearRegressionModel {
    pub W : Array1<f64>
}

impl LinearRegressionModel {
    pub fn _get_trained_variable(&self) -> &Array1<f64> {
        &self.W
    }

    // Gradiant descent for multiple features
    pub fn _fit(&mut self, X_train: Array2<f64>, Y_train: Array2<f64>, epoch: i32, alpha: f64, is_classification: bool) {
        let m  = X_train.nrows(); // number of training example
        let mf : f64 = m as f64; // number of training example as float 64
        let bias = Array::<f64, Ix2>::from_elem((X_train.shape()[0], 1), 1.);

        let X_train = concatenate![Axis(1), bias, X_train];

        for _ in 0..epoch {
            let mut Wbis = self.W.clone();
            for j in 0..(self.W.shape()[0]) {
                let mut w = Wbis[j];
                let mut grad = 0.;
                //println!("j : {},\n{:?}", j, X_train);

                for i in 0..m {
                    //getting the training example features [i]
                    let X_i = X_train.slice(s![i, ..]);

                    //getting the training example output
                    let Y_i = &Y_train.slice(s![i, ..]);

                    //Hypotesis function : WᵗX
                    let mut hyp = self.W.dot(&X_i);
                    //println!("«loop : {}..{} - {:?}\nxi: {:?}»",i, m ,  hyp, X_i);

                    if is_classification {
                        hyp = sig(&hyp);
                    }

                    let loss = hyp - Y_i;
                    grad += loss[[0]] * X_i[j];

                }
                //should implement the cost function W = W - (α / n) Σ (WᵗX - Y) * X
                // for each W at the same time
                Wbis[j] = w - (alpha / mf) * grad;
            }
            // updating W with new values
            self.W = Wbis;
            println!("new weight : {}", self.W)
        }
    }


    pub fn predict(&self, x_predict : Array1<f64>) -> f64 {
        let bias : Array1<f64> = array![1.0];
        let x = concatenate![Axis(0), bias, x_predict];

        self.W.dot(&x)
    }
}

fn sig(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}


fn main() {

    let X = get_csv2_f64(&String::from("../dataset/test_no_bias.txt")).unwrap();
    let (x1 , y1) = X.view().split_at(Axis(1), X.shape()[1] - 1);

    println!("X == {:?}", X);
    println!("{}", y1);

    let mut linear_r = LinearRegressionModel{ W: array![0.1, 0.1, 0.1] };
    linear_r._fit(x1.into_owned(), y1.into_owned(), 5000, 0.1, false);

    let result_w = linear_r.predict(array![0.7696284430248931,0.9606353623788391]);

    println!("{}", result_w);
}

pub fn get_csv2_f64(src : &String) -> Result<ArrayBase<OwnedRepr<f64>, Ix2>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(src)?;

    let mut data = Vec::new();
    let mut data2 = Vec::new();

    for result in rdr.records() {
        let record = result?;
        data.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>());
    }

    let ncol = data.first().map_or(0, |row| row.len());
    let mut nrows = 0;

    for i in 0..data.len() {
        data2.extend_from_slice(&data[i]);
        nrows += 1;
    }
    let array = Array2::from_shape_vec((nrows, ncol), data2).unwrap();
    Ok(array)
}
