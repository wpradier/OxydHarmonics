import ctypes
from typing import List
import numpy as np

# lib = ctypes.CDLL("your_rust_library.dll")  # Replace with the actual library filename
# lib = ctypes.CDLL("your_rust_library.so")  # Replace with the actual library filename

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/libml_core.so") # setup Etienne


ml_lib.create_rbf_model.argtypes  = [ctypes.c_int, ctypes.c_bool]
ml_lib.create_rbf_model.restype = ctypes.c_void_p

ml_lib.train_rbf_model.argtypes = train_rbf_argtypes = [
    ctypes.c_void_p,  # model_ptr
    ctypes.POINTER(ctypes.c_double),  # X_train
    ctypes.c_int,                     # X_rows
    ctypes.c_int,                     # X_cols
    ctypes.POINTER(ctypes.c_double),  # y_train
    ctypes.c_int,                     # y_rows
    ctypes.c_int,                     # y_cols
    ctypes.c_double,                  # lr
    ctypes.c_size_t                   # epochs
]
ml_lib.train_rbf_model.restype = None

ml_lib.predict_rbf_model.argtypes = predict_argtypes = [
    ctypes.c_void_p,                       # model
    ctypes.POINTER(ctypes.c_double),  # X
    ctypes.c_int32,                              # X_rows
    ctypes.c_bool                                 # is_classification
]

ml_lib.predict_rbf_model.restype = ctypes.POINTER(ctypes.c_double)

ml_lib.delete_rbf_model.argtypes = [ctypes.c_void_p]
ml_lib.delete_rbf_model.restype = None

ml_lib.delete_float_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
ml_lib.delete_float_array.restype = None



def create_rbf_model(arch: list, infer_stds: bool) -> int:

    model_pointer = ml_lib.create_rbf_model(arch, infer_stds)

    return model_pointer


def train_rbf_model(
    model: int,
    x_train: np.ndarray,
    lines: int,
    columns: int,
    y_train: List[int],
    y_train_columns: int,
    y_lines: int,
    alpha: float,
    epochs: int
):
    ml_lib.train_rbf_model(model,x_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) , lines, columns,
                                   y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     y_lines,y_train_columns, alpha, epochs )

    

    return None



def predict_rbf_model(model_pointer:int, sample_input:List[float],sample_inputs_size, is_classification:bool):

    prediction= ml_lib.predict_rbf_model(model_pointer, sample_input.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)),sample_inputs_size, is_classification)
    return np.ctypeslib.as_array(prediction, (1,))

def destroy_rbf_model(model_pointer:int):
    ml_lib.destroy_rbf_model(model_pointer)



def delete_float_array(prediction):
    ml_lib.delete_float_array(prediction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
# fontion who save the model and take the model pointer and the filename in parameter and return nothing

""" def save_rbf_model(model_pointer : int, filename:str ):
    ml_lib.save_rbf_model.argtypes = [
        ctypes.POINTER(model_pointer),
        ctypes.POINTER(filename.ctypes.c_char)
    ]

    ml_lib.save_rbf_model.restypes = None



def load_linear_model(filename):
    ml_lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    ml_lib.load_linear_model.restype = ctypes.POINTER(LinearRegressionModel)

    return ml_lib.load_linear_model(filename)
 """

# Load the Rust library using ctypes """
