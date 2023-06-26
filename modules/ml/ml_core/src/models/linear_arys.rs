use ndarray::{array, Array1, Array2, concatenate, Axis, Ix2, s, Array, ArrayBase, OwnedRepr};
use ndarray_rand::rand_distr::num_traits::abs;

#[allow(non_snake_case)]
pub struct LinearModelArys {
    pub W : Array1<f64>
}

impl LinearModelArys {
    pub fn _get_trained_variable(&self) -> &Array1<f64> {
        &self.W
    }

    // Gradiant descent for multiple features



    #[allow(non_snake_case)]
    pub fn _fit(&mut self,
                X_train: Array2<f64>,
                Y_train: Array2<f64>,
                epoch: i32,
                alpha: f64,
                is_classification: bool
    ) {

        //println!("ENTERING THE FIT {:?}, {:?}, {}, {}, {}", X_train, Y_train, epoch, alpha, is_classification);
        let m  = X_train.nrows(); // number of training example
        let mf : f64 = m as f64; // number of training example as float 64
        let bias = Array::<f64, Ix2>::from_elem((X_train.shape()[0], 1), 1.);


        let X_train = concatenate![Axis(1), bias, X_train];


        for _ in 0..epoch {
            let mut Wbis = self.W.clone();
            for j in 0..(self.W.shape()[0]) {
                let w = Wbis[j];
                let mut grad = 0.;
                //println!("j : {},\n{:?}", j, X_train);

                for i in 0..m {
                    //getting the training example features [i]
                    let X_i = X_train.slice(s![i, ..]);

                    //getting the training example output
                    let Y_i = &Y_train.slice(s![i, ..]);

                    //Hypotesis function : WᵗX
                    let mut hyp = self.W.dot(&X_i);
                    //println!("«loop : {}..{} -hypothesis {:?}»",i, m , hyp);

                    if is_classification {
                        hyp = sig(&hyp);
                    }

                    let loss = hyp - Y_i;
                    grad += loss[[0]] * X_i[j];
                    //println!("{}", loss[[0]]);
                }
                //should implement the cost function W = W - (α / n) Σ (WᵗX - Y) * X
                // for each W at the same time

                Wbis[j] = w - (alpha / mf) * grad;
            }
            // updating W with new values
            self.W = Wbis;
        }
        println!("new weight : {}", self.W);
    }


    pub fn predict(&self, x_predict : Array1<f64>, is_classification: bool) -> f64 {
        let bias : Array1<f64> = array![1.0];
        let x = concatenate![Axis(0), bias, x_predict];
        let res = self.W.dot(&x);

        if is_classification {
            if res >= 0.5 { 1. } else { 0. }
        }
        else {
            res
        }
    }

    pub fn test(&self, X_test : Array2<f64>, Y_test : Array2<f64>, pas : f64, is_classification : bool)
                -> f64{
        let m  = X_test.nrows(); // number of training example
        let y_flattened: Array2<f64> = Y_test.clone().into_shape((1, X_test.nrows())).unwrap();
        let Y_test= y_flattened.slice(s![0, ..]).into_owned();

        let mut res = 0.;
        for i in 0..m {
            let X_i = X_test.slice(s![i, ..]);

            //getting the training example output
            let Y_i = &Y_test[i];
            let y_pred = self.predict(X_i.into_owned(), is_classification);

            if abs(Y_i - y_pred) < pas {
                res += 1.;
            }
        }
        return (res / m as f64) * 100.
    }
}

fn sig(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_1() {
        let x: Array2<f64> = array![
        [1.0, 8.0],
        [4.0, 2.0],
        [5.0, 6.0]
    ];
        let y : Array2<f64> = array![
        [0.],
        [0.],
        [1.0]
    ];


        let mut model = LinearModelArys { W: array![0.1, 0.1, 0.1]};

        model._fit(x.clone(), y.clone(), 50000, 0.01, true);
        let res1 = model.predict(array![1., 8.], true);
        let test_result = model.test(x, y, 0.1, true, );

        println!("{}\n", test_result);

    }

    #[test]
     fn test_linear_regression_2() {
        let x: Array2<f64> = array![[1., 0.], [0., 1.], [0., 0.], [1., 1.]];
        let y : Array2<f64> = array![
            [1.],
            [1.],
            [0.],
            [0.]
        ];



        let mut model = LinearModelArys { W: array![0.1, 0.1, 0.1]};

        model._fit(x.clone(), y.clone(), 50000, 0.01, true);
        let res1 = model.predict(array![1., 0.], true);
        let test_result = model.test(x, y, 0.1, true, );
        println!("{}, {}", test_result, res1);

        assert_eq!(res1, 1.);

    }
}