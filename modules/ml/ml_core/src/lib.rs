use std::ffi::{c_char, CStr};
use crate::models::linear::{self, LinearRegressionModel};


mod models;


#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearRegressionModel {
    println!("IN CREATE");

    let usize_len = usize::try_from(len).unwrap();
    let model = Box::new(linear::create(usize_len));

    Box::leak(model)
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearRegressionModel,
                                 x_train: *const f64, lines: i32, columns: i32,
                                 y_train: *const f64, y_train_columns: i32,
                                    alpha: f64, epochs: i32, is_classification: bool) {

}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearRegressionModel, sample_input: *const f64, lines: i32) -> f64 {
    const V: f64 = 0.0;
    return V;
}

#[no_mangle]
extern "C" fn save_linear_model(model: *mut LinearRegressionModel, filename: *const c_char) {
    println!("IN SAVE");
    unsafe {
        println!("MODEL: {:?}", *model);
        let c_str = CStr::from_ptr(filename).to_str().unwrap();
        println!("FILENAME: {}", c_str);
    }
    println!("END SAVE");
}

#[no_mangle]
extern "C" fn destroy_linear_model(model: *mut LinearRegressionModel) {

}

#[no_mangle]
extern "C" fn load_linear_model(filename: *const u8) -> *mut LinearRegressionModel {
    let model = Box::new(linear::create(1));

    Box::leak(model)
}