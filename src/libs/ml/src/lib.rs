struct LinearModel {

}



#[no_mangle]
extern "C" fn create_linear_model() -> *mut LinearModel {
    let model = Box::new(LinearModel {});

    Box::leak(model)
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearModel, x_train: *const f64, lines: i32,
                                 columns: i32, y_train: *const f64, y_train_columns: i32,
                                    alpha: f64, epochs: i32) -> *const f64 {
    const V: f64 = 1.0;

    return &V;
}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearModel, sample_input: *const f64, lines: i32,
                                    columns: i32) -> f64 {
    const V: f64 = 0.0;
    return V;
}

#[no_mangle]
extern "C" fn save_linear_model(model: *mut LinearModel, filename: *const u8) {

}

#[no_mangle]
extern "C" fn destroy_linear_model(model: *mut LinearModel) {

}

#[no_mangle]
extern "C" fn load_linear_model(filename: *const u8) -> *mut LinearModel {
    let model = Box::new(LinearModel {});

    Box::leak(model)
}
