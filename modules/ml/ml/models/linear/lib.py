import ctypes
from typing import Optional, List
import numpy as np

# lib = ctypes.CDLL("../../../ml_core/target/debug/libml_core.so")  # Replace with the actual library filename
# lib = ctypes.CDLL("your_rust_library.so")  # Replace with the actual library filename

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/ml_core.dll")


def create_linear_model(length: int) -> int:
    ml_lib.create_linear_model.argtypes = [ctypes.c_int32]
    ml_lib.create_linear_model.restype = ctypes.c_void_p

    return ml_lib.create_linear_model(length)


# #[no_mangle]
# extern "C" fn train_linear_model(model: *mut LinearRegressionModel,
#                                  x_train: *const f64, lines: i32, columns: i32,
#                                  y_train: *const f64, y_train_columns: i32,
#                                     alpha: f64, epochs: i32, is_classification: bool) {
# }

# TODO inflate x_train (and y_train)
def train_linear_model(
        model: int,
        x_train: np.ndarray,
        lines: int,
        columns: int,
        y_train: np.ndarray,
        y_train_columns: int,
        alpha: float,
        epochs: int,
        is_classification: bool
):
    ml_lib.train_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_bool
    ]
    ml_lib.train_linear_model.restype = None

    ml_lib.train_linear_model(
        model,
        x_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lines,
        columns,
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_train_columns,
        alpha,
        epochs,
        is_classification
    )


def predict_linear_model(
        model: int,
        sample_input: np.ndarray,
        lines: int,
        is_classification: bool
):
    ml_lib.predict_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_bool
    ]
    ml_lib.predict_linear_model.restype = ctypes.c_double

    res = ml_lib.predict_linear_model(
        model,
        sample_input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lines,
        is_classification
    )

    return res


def test_linear_model(
        model: int,
        x_test: np.ndarray,
        lines: int,
        columns: int,
        y_test: np.ndarray,
        y_test_columns: int,
        pas: float,
        is_classification: bool
):
    ml_lib.test_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_bool
    ]
    ml_lib.test_linear_model.restype = ctypes.c_double
    print(x_test)

    return ml_lib.test_linear_model(
        model,
        x_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lines,
        columns,
        y_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_test_columns,
        pas,
        is_classification
    )


def save_linear_model(
        model: int,
        filename: str
):
    ml_lib.save_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]

    ml_lib.save_linear_model.restype = None
    file_cstr = ctypes.c_char_p(filename.encode("utf-8"))

    print(file_cstr)
    print(type(file_cstr))

    ml_lib.save_linear_model(
        model,
        file_cstr
    )


"""

def destroy_linear_model(model):
    lib.destroy_linear_model.argtypes = [ctypes.POINTER(LinearRegressionModel)]
    lib.destroy_linear_model.restypes = None


def load_linear_model(filename):
    lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.load_linear_model.restype = ctypes.POINTER(LinearRegressionModel)

    return lib.load_linear_model(filename)


# Load the Rust library using ctypes
"""


def loadDataSet(path):
    DATA_train = np.genfromtxt(path, delimiter=',')
    X = DATA_train[:, 1:]
    Y = DATA_train[:, :1]
    X_size = len(X)
    X_row_size = len(X[0])
    all_training_inputs = np.float64(X.flatten())
    all_training_expected_outputs = np.float64(Y.flatten())
    return all_training_inputs, all_training_expected_outputs, X_size, X_row_size
