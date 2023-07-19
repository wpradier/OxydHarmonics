use std::alloc::{dealloc, Layout};
use std::ffi::{c_char, c_int, c_uint, CStr};
use std::ops::Deref;
use std::ptr::slice_from_raw_parts;
use std::result;
use crate::models::mlp::{self, MultilayerPerceptron};


mod models;
mod utils;


/** MULTILAYER PERCEPTRON **/

#[no_mangle]
extern "C" fn create_mlp_model(structure: *mut usize, len: usize) -> *mut MultilayerPerceptron {
    unsafe {
        let mlp_structure = slice_from_raw_parts(structure, usize::try_from(len).unwrap())
            .as_ref().unwrap().to_vec();


        let model = Box::new(mlp::create(mlp_structure));

        Box::leak(model)
    }
}


#[no_mangle]
extern "C" fn train_mlp_model(model: *mut MultilayerPerceptron,
                              x_train: *mut f64, lines: i32, columns: i32,
                              y_train: *mut f64, y_lines: i32, y_columns: i32,
                              alpha: f64, epochs: u32, is_classification: bool,
                              train_name_c: *const c_char) {
    unsafe {
        let u_lines = usize::try_from(lines).unwrap();
        let u_columns = usize::try_from(columns).unwrap();
        let u_y_lines = usize::try_from(y_lines).unwrap();
        let u_y_columns = usize::try_from(y_columns).unwrap();

        let input_dataset = slice_from_raw_parts(x_train, u_lines * u_columns)
            .as_ref().unwrap()
            .chunks(u_columns)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        let output_dataset = slice_from_raw_parts(y_train, u_y_lines * u_y_columns)
            .as_ref().unwrap()
            .chunks(u_y_columns)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        //println!("INPUT VEC: {:?}", input_dataset);
        //println!("OUTPUT VEC: {:?}", output_dataset);

        let train_name = CStr::from_ptr(train_name_c).to_str().unwrap();

        mlp::train(model.as_mut().unwrap(), &input_dataset, &output_dataset, alpha, epochs, is_classification, train_name);
    }
}

#[no_mangle]
extern "C" fn predict_mlp_model(model: *mut MultilayerPerceptron, sample_input: *mut f64, columns: i32, is_classification: bool) -> *mut [f64] {
    unsafe {
        let input_vec = slice_from_raw_parts(sample_input, usize::try_from(columns).unwrap()).as_ref().unwrap().to_vec();

        let prediction = mlp::predict(model.as_mut().unwrap(), &input_vec, is_classification);

        Box::<[f64]>::into_raw(prediction.into_boxed_slice())
    }
}

#[no_mangle]
extern "C" fn destroy_mlp_model(model: *mut MultilayerPerceptron) {
    unsafe {
        dealloc(model as *mut u8, Layout::new::<MultilayerPerceptron>());
    }
}

#[no_mangle]
extern "C" fn save_mlp_model(model: *mut MultilayerPerceptron, filename: *const c_char) {
    unsafe {
        let c_str = CStr::from_ptr(filename).to_str().unwrap();

        mlp::save(model.as_ref().unwrap(), c_str).unwrap()
    }
}

#[no_mangle]
extern "C" fn load_mlp_model(filename: *const c_char) -> *mut MultilayerPerceptron {
    unsafe {
        let c_str = CStr::from_ptr(filename).to_str().unwrap();
        let model = mlp::load(c_str).unwrap();
        let boxed_model = Box::new(model);

        Box::leak(boxed_model)
    }

}