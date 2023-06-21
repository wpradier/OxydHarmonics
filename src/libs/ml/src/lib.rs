//mod pmc_model;
mod linear_model;

//use pmc_model::{create_pmc,train_pmc , PmcModel};
use linear_model::{LinearRegressionModel};
use std::slice;
use ndarray::{Array, Array1, Array2, ArrayView};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use tensorboard_rs;



#[no_mangle]
extern "C" fn create_linear_model(len: i32) -> *mut LinearRegressionModel {
    unsafe {
        let model_size: usize = len as usize;
        let layer_weight = Array1::random(
            model_size,
            Uniform::new(-1.0, 1.0)
        );
        let model = Box::new(LinearRegressionModel { W: layer_weight });

        Box::leak(model)
    }
}


#[no_mangle]
extern "C" fn train_linear_model(model: *mut LinearRegressionModel,
                                 x_train: *const f64, lines: i32, columns: i32,
                                 y_train: *const f64, y_train_columns: i32,
                                    alpha: f64, epochs: i32, is_classification: bool){
   let mut linear_model : LinearRegressionModel =
       unsafe {
           model.read()
       };

    let x_train_slice = unsafe {
        slice::from_raw_parts(x_train, (lines * columns) as usize)
    };

    let y_train_slice = unsafe {
        slice::from_raw_parts(y_train, y_train_columns as usize)
    };

    let _x_train = Array2::from_shape_vec((lines as usize, columns as usize), x_train_slice.to_vec()).unwrap();
    let _y_train = Array2::from_shape_vec((1, y_train_columns as usize), y_train_slice.to_vec()).unwrap();

    linear_model._fit(_x_train, _y_train, epochs, alpha, is_classification);

    /*
    let mut _x_train : Array2<f64> =
        unsafe {
            Array2::from_vec(Vec::from_raw_parts(x_train.cast_mut(),
                                                 (columns * lines) as usize,
                                                 (columns * lines) as usize))
                .reshape((lines, columns))
        };

    let mut _y_train : ArrayView<f64, _> =
        unsafe {
            ArrayView::from_shape_ptr((1, y_train_columns), y_train).clone()
            /*
            Array2::from_vec(Vec::from_raw_parts(y_train.cast_mut(),
                                                 y_train_columns as usize,
                                                 (y_train_columns + 1) as usize))
                .reshape(1, y_train_columns)

             */
        };
    //let x : Array2<f64> = _y_train.clone();
     */


}

#[no_mangle]
extern "C" fn predict_linear_model(model: *mut LinearModel, sample_input: *const f64, lines: i32) -> f64 {
    let mut linear_model : LinearRegressionModel =
        unsafe {
            model.read()
        };

    let slice_sample = unsafe {
        slice::from_raw_parts(sample_input, lines as usize)
    };

    let _sample_input = Array1::from_shape_vec((lines as usize, 1), slice_sample.to_vec()).unwrap();



    return linear_model.predict(_sample_input);
}
/*

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
 */