mod pmc_model;
use pmc_model::{create_pmc,train_pmc ,predict_pmc, PmcModel};
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
pub extern "C" fn save_mlp_model(model: *mut PmcModel, filename: *const c_char) {
    unsafe {
        let model_ref = &*model;
        let filename_cstr = CStr::from_ptr(filename);
        let filename_str = filename_cstr.to_str().expect("Invalid UTF-8 filename");

        // Serialize the model to JSON
        let serialized_model = serde_json::to_string(model_ref).expect("Failed to serialize the model");

        // Save the serialized model to the file
        let mut file = File::create(filename_str).expect("Failed to create the file");
        file.write_all(serialized_model.as_bytes()).expect("Failed to write the serialized model to the file");

        println!("MLP model saved to file: {}", filename_str);
    }
}

#[no_mangle]
pub extern "C" fn load_mlp_model(filename: *const c_char) -> *mut PmcModel {
    unsafe {
        let filename_cstr = CStr::from_ptr(filename);
        let filename_str = filename_cstr.to_str().expect("Invalid UTF-8 filename");

        // Load the serialized model from the file
        let mut file = File::open(filename_str).expect("Failed to open the file");
        let mut serialized_model = String::new();
        file.read_to_string(&mut serialized_model).expect("Failed to read the serialized model from the file");

        // Deserialize the model from JSON
        let deserialized_model: PmcModel = serde_json::from_str(&serialized_model).expect("Failed to deserialize the model");

        // Allocate memory for the model and copy the deserialized model into it
        let model_ptr = Box::into_raw(Box::new(deserialized_model));

        println!("MLP model loaded from file: {}", filename_str);

        model_ptr
    }
}
