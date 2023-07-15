mod pmc_model;
use pmc_model::{create_pmc,train_pmc ,predict_pmc, PmcModel, save_pmc, load_pmc};
use std::slice;
use std::os::raw::c_char;
use std::ffi::CStr;
use std::fs::File;
use std::io::{Read, Write};

use serde_json;





#[no_mangle]
extern "C" fn create_mlp_model(npl: *mut i32, npl_len: i32) -> *mut PmcModel {
    println!("create_mlp_model");
    unsafe {
        let ptr: *mut usize = npl as *mut usize;
        let len: usize = npl_len as usize;
        let slice: &[usize] = slice::from_raw_parts(ptr, len);
        let vec: Vec<usize> = slice.to_vec();
        let model: PmcModel;
        model = create_pmc(vec);
        let val_d = model.d;
        let val_l = model.L;
        let val_w = model.W;
        let val_x = model.X;
        let val_deltas = model.deltas;

        // Créer la boîte contenant le modèle
        let output_model = Box::new(PmcModel { d: val_d, L: val_l, W: val_w, X: val_x, deltas: val_deltas });

        let mymodel = Box::leak(output_model);
        println!("create_mlp_model done");

        mymodel
    }
}

#[no_mangle]
extern "C" fn train_mlp_model(model_ptr: *mut PmcModel, X_train: *const f64, lines: i32,
                              columns: i32, y_train: *const f64, output_columns: i32,
                              alpha: f64, num_iter: i32, is_classification: bool) {
    println!("train_mlp_model");
    unsafe {
        let mut model = model_ptr.as_mut().unwrap();
        
        let X_train = std::slice::from_raw_parts(X_train, (lines * columns) as usize)
            .chunks(columns as usize)
            .map(|chunk| {
                let mut input = Vec::with_capacity(columns as usize);
                input.extend_from_slice(chunk);
                input
            })
            .collect::<Vec<Vec<f64>>>();

        let y_train = std::slice::from_raw_parts(y_train, (lines * output_columns) as usize)
            .chunks(output_columns as usize)
            .map(|chunk| {
                let mut output = Vec::with_capacity(output_columns as usize);
                output.extend_from_slice(chunk);
                output
            })
            .collect::<Vec<Vec<f64>>>();

        train_pmc(&mut model, X_train, y_train, is_classification, num_iter, alpha);
        println!("train_mlp_model done");
    
}
                              }


#[no_mangle]
extern "C" fn predict_mlp_model(model: *mut PmcModel, sample_inputs: *const f64, len_columns: usize, is_classification: bool) -> *mut f64 {
    unsafe {
        let model_ref = model.as_mut().unwrap();
        let inputs: Vec<f64> = {
            let input_slice = std::slice::from_raw_parts(sample_inputs, len_columns);
            input_slice.to_vec()
        };
        let prediction = predict_pmc(model_ref, inputs, is_classification);
        let fake_output: Vec<f64> = prediction;
        println!("preditiction {:?}", fake_output);

        Box::<[f64]>::into_raw(fake_output.into_boxed_slice()) as *mut f64

    }
}


#[no_mangle]
extern "C" fn delete_mlp_model(model: *mut PmcModel) {
    unsafe {
        //Box::from_raw(model);
        drop(Box::from_raw(model))
    }
}

#[no_mangle]
extern "C" fn delete_float_array(arr: *mut f32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}


#[no_mangle]
extern "C" fn load_mlp_model(filename: *const c_char) -> *mut PmcModel {
    unsafe {
        let filename_cstr = CStr::from_ptr(filename);
        let filename_str = match filename_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let model = match load_pmc(filename_str) {
            Ok(m) => m,
            Err(_) => return std::ptr::null_mut(),
        };
        Box::into_raw(Box::new(model))
    }
}

#[no_mangle]
extern "C" fn save_mlp_model(model: *mut PmcModel, filename: *const c_char) -> bool {
    unsafe {
        let model_ref = model.as_ref().unwrap();
        let filename_cstr = CStr::from_ptr(filename);
        let filename_str = match filename_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if let Err(_) = save_pmc(model_ref, filename_str) {
            return false;
        }
    }
    true
}