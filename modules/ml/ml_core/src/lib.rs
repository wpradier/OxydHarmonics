mod rbf_net;
use rbf_net::{generate_random_weights, RBFNet,train_rbf,predict_rbf,create_model};



#[no_mangle]
pub extern "C" fn create_rbf_model(arch: *const usize, arch_len: i32, infer_stds: bool) -> *mut RBFNet {
    // Convert the `arch` pointer to a slice
    let arch_slice = unsafe { std::slice::from_raw_parts(arch, arch_len as usize) };

    let model = create_model(arch_slice);

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

        train_rbf(model_ref, &x_train_slice, &y_train_slice, alpha, epochs);
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
        let prediction = predict_rbf(model_ref, &[inputs], is_classification);

        let boxed_slice = prediction.into_boxed_slice();
        let raw_ptr = Box::into_raw(boxed_slice) as *mut f64;

        raw_ptr
    }
}

#[no_mangle]
pub extern "C" fn delete_rbf_model(model: *mut RBFNet) {
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

