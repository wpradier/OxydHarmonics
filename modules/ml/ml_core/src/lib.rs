mod models;

use std::slice;
use ndarray::{Array, array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use tensorboard_rs;
use crate::models::linear_arys::LinearModelArys;


#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearModelArys{
        let model_size: usize = len as usize;
        let layer_weight = Array1::random(
            model_size,
            Uniform::new(-1.0, 1.0)
        );
        let model = Box::new(LinearModelArys{ W: layer_weight });

        Box::leak(model)
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearModelArys,
                                 x_train: *const f64, lines: i32, columns: i32,
                                 y_train: *const f64, y_train_columns: i32,
                                 alpha: f64, epochs: i32, is_classification: bool){
    unsafe {
        let mut linear_model : &mut LinearModelArys = model.as_mut().unwrap();

        let x_train_slice = unsafe {
            slice::from_raw_parts(x_train, (lines * columns) as usize)
        };

        let y_train_slice = unsafe {
            slice::from_raw_parts(y_train, y_train_columns as usize)
        };

        let _x_train = Array2::from_shape_vec((lines as usize, columns as usize), x_train_slice.to_vec()).unwrap();
        let _y_train = Array2::from_shape_vec((y_train_columns as usize, 1), y_train_slice.to_vec()).unwrap();

        linear_model._fit(_x_train, _y_train, epochs, alpha, is_classification);

    }

}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearModelArys, sample_input: *const f64, lines: i32) -> f64 {
    unsafe {
        let linear_model: &mut LinearModelArys = model.as_mut().unwrap();

        let slice_sample = unsafe {
            slice::from_raw_parts(sample_input, lines as usize)
        };

        let _sample_input = Array1::from_shape_vec((lines as usize), slice_sample.to_vec()).unwrap();


        println!(" predict : {:?}", _sample_input);
        return linear_model.predict(_sample_input);
    }

}

#[no_mangle]
extern "C" fn save_linear_model(model: *mut LinearModelArys, filename: *const u8) {

}

#[no_mangle]
extern "C" fn destroy_linear_model(model: *mut LinearModelArys) {

}

#[no_mangle]
extern "C" fn load_linear_model(filename: *const u8) -> *mut LinearModelArys {
    let model = Box::new(LinearModelArys { W: array![1.] });

    Box::leak(model)
}
