use ndarray::{array, Array1, Array2, ArrayView, concatenate, Axis, Ix2, s, Array};

pub struct LinearModelArys {
    pub W : Array1<f64>
}

impl LinearModelArys {
    pub fn _get_trained_variable(&self) -> &Array1<f64> {
        &self.W
    }

    // Gradiant descent for multiple features
    pub fn _fit(&mut self, X_train: Array2<f64>, Y_train: Array2<f64>, epoch: i32, alpha: f64, is_classification: bool) {

        println!("ENTERING THE FIT {:?}, {:?}, {}, {}, {}", X_train, Y_train, epoch, alpha, is_classification);
        let m  = X_train.nrows(); // number of training example
        let mf : f64 = m as f64; // number of training example as float 64
        let bias = Array::<f64, Ix2>::from_elem((X_train.shape()[0], 1), 1.);

        let X_train = concatenate![Axis(1), bias, X_train];

        for _ in 0..epoch {
            let mut Wbis = self.W.clone();
            for j in 0..(self.W.shape()[0]) {
                let mut w = Wbis[j];
                let mut grad = 0.;
                println!("j : {},\n{:?}", j, X_train);

                for i in 0..m {
                    //getting the training example features [i]
                    let X_i = X_train.slice(s![i, ..]);

                    //getting the training example output
                    let Y_i = &Y_train.slice(s![i, ..]);

                    //Hypotesis function : WᵗX
                    let mut hyp = self.W.dot(&X_i);
                    println!("«loop : {}..{} -hypothesis {:?}\nxi: {:?}»",i, m ,  hyp, X_i);

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
        println!("FINISHING FIT")
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