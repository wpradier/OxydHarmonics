use std::alloc::{dealloc, Layout};
use std::ffi::{c_char, c_int, c_uint, CStr};
use std::slice;
use std::ops::Deref;
use std::ptr::slice_from_raw_parts;
use std::result;
use ndarray::{array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::models::linear::LinearModel;
use crate::models::mlp::{self, MultilayerPerceptron};
use crate::models::rbf::{self, RBFNet};

mod models;
mod utils;

/** LINEAR **/
#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearModel {
        let model_size: usize = len as usize;
        let layer_weight = Array1::random(
            model_size,
            Uniform::new(-1.0, 1.0)
        );
        let model = Box::new(LinearModel { W: layer_weight });

        Box::leak(model)
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearModel,
                                 x_train: *const f64, lines: i32, columns: i32,
                                 y_train: *const f64, y_train_columns: i32,
                                 alpha: f64, epochs: i32, is_classification: bool){
    unsafe {
        let linear_model : &mut LinearModel = model.as_mut().unwrap();

        let x_train_slice = slice::from_raw_parts(x_train, (lines * columns) as usize);

        let y_train_slice = slice::from_raw_parts(y_train, y_train_columns as usize);

        let _x_train = Array2::from_shape_vec((lines as usize, columns as usize), x_train_slice.to_vec()).unwrap();
        let _y_train = Array2::from_shape_vec((y_train_columns as usize, 1), y_train_slice.to_vec()).unwrap();

        linear_model._fit(_x_train, _y_train, epochs, alpha, is_classification);

    }

}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearModel, sample_input: *const f64, lines: i32,
                                   is_classification: bool) -> f64 {
    unsafe {
        let linear_model: &mut LinearModel = model.as_mut().unwrap();

        let slice_sample = slice::from_raw_parts(sample_input, lines as usize);

        let _sample_input = Array1::from_shape_vec(lines as usize, slice_sample.to_vec()).unwrap();


        println!(" predict : {:?}", _sample_input);
        return linear_model.predict(_sample_input, is_classification);
    }

}

#[no_mangle]
extern "C" fn test_linear_model(model: *mut LinearModel,
                                x_test: *const f64, lines: i32, columns: i32,
                                y_test: *const f64, y_columns: i32,
                                pas: f64, is_classification: bool) -> f64 {
    unsafe {
        let linear_model: &mut LinearModel = model.as_mut().unwrap();

        let x_test_slice = slice::from_raw_parts(x_test, (lines * columns) as usize);

        let y_test_slice = slice::from_raw_parts(y_test, y_columns as usize);

        println!("column : {}\n row : {}\n{:?}", columns, lines, x_test_slice);


        let _x_test = Array2::from_shape_vec((lines as usize, columns as usize), x_test_slice.to_vec()).unwrap();

        println!("{:?}", _x_test);
        let _y_test = Array2::from_shape_vec((y_columns as usize, 1), y_test_slice.to_vec()).unwrap();

        return linear_model.test(_x_test, _y_test, pas, is_classification);
    }


}

#[no_mangle]
extern "C" fn save_linear_model(_model: *mut LinearModel, _filename: *const c_char) {
    unsafe {
        println!("enter save_linear");

        let linear_model: &mut LinearModel = _model.as_mut().unwrap();

        let new_path = CStr::from_ptr(_filename);

        let path_str = new_path.to_str().expect("Invalid UTF-8 filename").to_owned();
        let path = String::from(path_str);
        println!("path : {}", path);

        linear_model.save(path).expect("failed load? or success");
    }

}

#[no_mangle]
extern "C" fn load_linear_model(_filename: *const c_char) -> *mut LinearModel {
    unsafe {

        let new_path = CStr::from_ptr(_filename);
        let model = LinearModel::load(new_path.to_string_lossy().into_owned()).unwrap() ;
        Box::leak(model)
    }

}

#[no_mangle]
extern "C" fn destroy_linear_model(_model: *mut LinearModel) {
    unsafe {
        dealloc(model as *mut u8, Layout::new::<LinearModel>());
    }
}

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

/** RBF **/

#[no_mangle]
pub extern "C" fn create_rbf_model(arch: *const usize, arch_len: i32, infer_stds: bool) -> *mut RBFNet {
    // Convert the `arch` pointer to a slice
    let arch_slice = unsafe { std::slice::from_raw_parts(arch, arch_len as usize) };

    let model = rbf::create(arch_slice);

    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn train_rbf_model(
    model: *mut RBFNet,
    x_train: *const f64,
    lines: i32,
    columns: i32,
    y_train: *const f64,
    y_rows: i32,
    y_cols: i32,
    alpha: f64,
    epochs: usize,
) {
    unsafe {
        let model_ref = model.as_mut().unwrap();

        let x_train_slice = std::slice::from_raw_parts(x_train, (lines * columns) as usize)
            .chunks(columns as usize)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<Vec<f64>>>();

        let y_train_slice = std::slice::from_raw_parts(y_train, (y_rows * y_cols) as usize)
            .chunks(y_cols as usize)
            .map(|chunk| chunk[0])
            .collect::<Vec<f64>>();

        rbf::train(model_ref, &x_train_slice, &y_train_slice, alpha, epochs);
    }
}

#[no_mangle]
pub extern "C" fn predict_rbf_model(
    model: *mut RBFNet,
    sample_inputs: *const f64,
    len_columns: usize,
    is_classification: bool,
) -> *mut f64 {
    unsafe {
        let model_ref = model.as_mut().unwrap();
        let inputs_slice = std::slice::from_raw_parts(sample_inputs, len_columns);

        let inputs = inputs_slice.to_vec();
        let prediction = rbf::predict(model_ref, &[inputs], is_classification);

        let boxed_slice = prediction.into_boxed_slice();
        let raw_ptr = Box::into_raw(boxed_slice) as *mut f64;

        raw_ptr
    }
}

#[no_mangle]
pub extern "C" fn destroy_rbf_model(model: *mut RBFNet) {
    if !model.is_null() {
        unsafe {
            Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn delete_float_array(arr: *mut f64, arr_len: i32) {
    if !arr.is_null() {
        unsafe {
            Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize);
        }
    }
}

