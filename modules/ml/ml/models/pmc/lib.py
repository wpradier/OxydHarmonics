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

ml_lib.save_mlp_model.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_char)]
ml_lib.save_mlp_model.restype = None

ml_lib.load_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
ml_lib.load_mlp_model.restype = ctypes.c_void_p


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



def save_mlp_model(model_ptr : int, filename: str):
    model_ptr = ctypes.c_void_p(model_ptr)
    filename_cstr = ctypes.c_char_p(filename.encode("utf-8"))
    ml_lib.save_mlp_model(model_ptr, filename_cstr)



def load_mlp_model(filename):
    # Convertir le nom de fichier en une chaîne de caractères C
    filename_cstr = ctypes.c_char_p(filename.encode("utf-8"))

    # Appeler la fonction Rust load_mlp_model
    model_ptr = ml_lib.load_mlp_model(filename_cstr)

    # Retourner le pointeur vers le modèle
    return model_ptr
