import ctypes
from typing import Optional, List
import numpy as np

# lib = ctypes.CDLL("your_rust_library.dll")  # Replace with the actual library filename
# lib = ctypes.CDLL("your_rust_library.so")  # Replace with the actual library filename

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/libml_core.dylib") # setup Etienne


ml_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
ml_lib.create_mlp_model.restype = ctypes.c_void_p

ml_lib.train_mlp_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
                                   ctypes.c_bool]
ml_lib.train_mlp_model.restype = None

ml_lib.predict_mlp_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_bool]
ml_lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double)

ml_lib.delete_mlp_model.argtypes = [ctypes.c_void_p]
ml_lib.delete_mlp_model.restype = None

ml_lib.delete_float_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
ml_lib.delete_float_array.restype = None



def create_mlp_model(npl: np.array , npl_size : int) -> int:

    model_pointer = ml_lib.create_mlp_model(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), npl_size)

    return model_pointer


def train_mlp_model(
        model: int,
        x_train: np.ndarray,
        lines: int,
        columns: int,
        y_train: List[int],
        y_train_columns: int,
        alpha: float,
        epochs: int,
        is_classification: bool
):
    ml_lib.train_mlp_model(model,x_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) , lines, columns,
                                   y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y_train_columns, alpha, epochs ,is_classification
    )

    return None


def predict_mlp_model(model_pointer:int, sample_input:List[int],sample_inputs_size, is_classification:bool, npl : List):
    prediction= ml_lib.predict_mlp_model(model_pointer, sample_input.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)),sample_inputs_size, is_classification)
    return np.ctypeslib.as_array(prediction, (npl[-1],))

def delete_mlp_model(model_pointer:int):
    ml_lib.delete_mlp_model(model_pointer)



def delete_float_array(prediction, npl : List):
    ml_lib.delete_float_array(prediction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), npl[-1])
# fontion who save the model and take the model pointer and the filename in parameter and return nothing

""" def save_mlp_model(model_pointer : int, filename:str ):
    ml_lib.save_mlp_model.argtypes = [
        ctypes.POINTER(model_pointer),
        ctypes.POINTER(filename.ctypes.c_char)
    ]

    ml_lib.save_mlp_model.restypes = None



def load_linear_model(filename):
    ml_lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    ml_lib.load_linear_model.restype = ctypes.POINTER(LinearRegressionModel)

    return ml_lib.load_linear_model(filename)
 """

# Load the Rust library using ctypes """
