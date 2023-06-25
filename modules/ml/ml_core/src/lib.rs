mod models;

use std::slice;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use tensorboard_rs;
use crate::models::linear_arys::LinearModelArys;


#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearModelArys{
    unsafe {
        let model_size: i32 = len;
        let layer_weight = Array::random(
            (1, model_size + 1),
            Uniform::new(-1.0, 1.0)
        );
        let model = Box::new(LinearModelArys { W: layer_weight });

        Box::leak(model)
    }
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearModelArys,
                                 x_train: *const f64, lines: i32, columns: i32,
                                 y_train: *const f64, y_train_columns: i32,
                                    alpha: f64, epochs: i32, is_classification: bool){
   let mut linear_model : LinearModelArys =
       unsafe {
           model.read()
       };

    let mut _x_train : Array2<f64> =
        unsafe {
            Array2::from_vec(Vec::from_raw_parts(x_train.cast_mut(),
                                                 (columns * lines) as usize,
                                                 (columns * lines) as usize))
                .reshape((lines, columns))
        };

    let mut _y_train : Array2<f64> =
        unsafe {
            Array2::from_vec(Vec::from_raw_parts(y_train.cast_mut(),
                                                 y_train_columns as usize,
                                                 (y_train_columns + 1) as usize))
                .reshape(1, y_train_columns)
        };

    linear_model._fit(_x_train, _y_train, epochs, alpha, is_classification);
}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearModelArys, sample_input: *const f64, lines: i32) -> f64 {
     let mut linear_model : LinearModelArys =
       unsafe {
           model.read()
       };

    let mut x_input : Array1<f64> =
        unsafe {
            Array1::from_vec(Vec::from_raw_parts(sample_input.cast_mut(),
                                                 (lines) as usize,
                                                 (lines) as usize))
        };

    return linear_model.predict(x_input);
}

#[no_mangle]
extern "C" fn save_linear_model(model: *mut LinearModelArys, filename: *const u8) {

}

#[no_mangle]
extern "C" fn destroy_linear_model(model: *mut LinearModelArys) {

}

#[no_mangle]
extern "C" fn load_linear_model(filename: *const u8) -> *mut LinearModelArys {
    let model = Box::new(LinearModelArys { W: () });

    Box::leak(model)
}
