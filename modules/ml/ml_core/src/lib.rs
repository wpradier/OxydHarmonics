use std::alloc::{dealloc, Layout};
use std::ffi::{c_char, CStr};
use std::ptr::slice_from_raw_parts;
use std::result;
use crate::models::linear::{self, LinearRegressionModel};


mod models;


#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearRegressionModel {
    let usize_len = usize::try_from(len).unwrap();
    let model = Box::new(linear::create(usize_len));

    Box::leak(model)
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearRegressionModel,
                                 x_train: *mut f64, lines: usize, columns: usize,
                                 y_train: *mut f64, y_train_columns: usize,
                                    alpha: f64, epochs: u32, is_classification: bool) {
    unsafe {
        let input_dataset = slice_from_raw_parts(x_train, lines * columns)
            .as_ref().unwrap()
            .chunks(columns)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        let output_dataset = Vec::from_raw_parts(y_train, y_train_columns, y_train_columns);

        println!("INPUT VEC: {:?}", input_dataset);
        println!("OUTPUT VEC: {:?}", output_dataset);


        linear::train(model.as_mut().unwrap(), input_dataset, output_dataset, alpha, epochs, is_classification);
    }
}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearRegressionModel, sample_input: *mut f64, columns: usize, is_classification: bool) -> f64 {
    unsafe {
        let input_vec = Vec::from_raw_parts(sample_input, columns, columns + 1);

        linear::predict(model.as_ref().unwrap(), &input_vec, is_classification)
    }
}

#[no_mangle]
extern "C" fn save_linear_model(model: *mut LinearRegressionModel, filename: *const c_char) {
    unsafe {
        let c_str = CStr::from_ptr(filename).to_str().unwrap();
        //TODO SAVE
    }
}

#[no_mangle]
extern "C" fn destroy_linear_model(model: *mut LinearRegressionModel) {
    unsafe {
        dealloc(model as *mut u8, Layout::new::<LinearRegressionModel>());
    }
}

#[no_mangle]
extern "C" fn load_linear_model(filename: *const u8) -> *mut LinearRegressionModel {
    let model = Box::new(linear::create(1));

    Box::leak(model)
}