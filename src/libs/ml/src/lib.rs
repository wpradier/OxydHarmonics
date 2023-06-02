mod pmc_model;
use pmc_model::{create_pmc,train_pmc , PmcModel};
use std::slice;
use tensorboard_rs;
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


#[no_mangle]
extern "C" fn create_mlp_model(npl: *mut i32, npl_len: usize) -> *mut PmcModel {
    unsafe {
        let ptr: *mut i32 = npl;
        let len: usize = npl_len;
        let slice: &[i32] = slice::from_raw_parts(ptr, len);
        let vec: Vec<i32> = slice.to_vec();
        let model: PmcModel;
        unsafe {
            model = create_pmc(vec);
        }
        let val_d = model.d;
        let val_l = model.L;
        let val_w = model.W;
        let val_x = model.X;
        let val_deltas = model.deltas;

        // Créer la boîte contenant le modèle
        let output_model = Box::new(PmcModel { d: val_d, L: val_l, W: val_w, X: val_x, deltas: val_deltas });

        Box::into_raw(output_model)
    }
}

#[no_mangle]
extern "C" fn train_pmc_model(model: *mut PmcModel, X_train: *const f64, lines: i32,
                                 columns: i32, y_train: *const f64, y_train_columns: i32,
                                    alpha: f64, epochs: i32) -> *const f64 {
                           unsafe{
                            let model_ptr = model;
                            let X_train_ptr = X_train;
                            

                           }             
    const V: f64 = 1.0;


    return &V;
}

#[no_mangle]
extern "C" fn delete_pmc_model(model: *mut PmcModel) {
    unsafe {
        Box::from_raw(model);
    }
}
