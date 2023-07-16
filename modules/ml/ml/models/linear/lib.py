import ctypes
from typing import Optional
import numpy as np

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/ml_core.dll")


def create_linear_model(length: int) -> Optional[int]:
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
        y_train: np.ndarray,
        alpha: float,
        epochs: int,
        is_classification: bool
):

    c_x_train = x_train.astype(float).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_y_train = y_train.astype(float).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

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

    return ml_lib.train_linear_model(
        model,
        c_x_train,
        len(x_train),
        len(x_train[0]),
        c_y_train,
        len(y_train),
        alpha,
        epochs,
        is_classification
    )


def predict_linear_model(model: int, sample_input: np.ndarray, is_classification: bool):
    ml_lib.predict_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_bool
    ]
    ml_lib.predict_linear_model.restype = ctypes.c_double

    c_sample = sample_input.astype(float).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result = ml_lib.predict_linear_model(
        model,
        c_sample,
        len(sample_input),
        is_classification
    )

    return result


def save_linear_model(model: int, filename: bytes):
    ml_lib.save_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p
    ]

    ml_lib.save_linear_model.restypes = None

    ml_lib.save_linear_model(model, filename)


def destroy_linear_model(model: int):
    ml_lib.destroy_linear_model.argtypes = [ctypes.c_void_p]
    ml_lib.destroy_linear_model.restypes = None

    return ml_lib.destroy_linear_model(model)


def load_linear_model(filename):
    lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.load_linear_model.restype = ctypes.POINTER(LinearRegressionModel)

    return lib.load_linear_model(filename)


# Load the Rust library using ctypes

if __name__ == '__main__':
    print("STARTING IN PYTHON")
    X = np.array([
        [1., 1.],
        [2., 3.],
        [3., 3.],
        [6., 4.],
        [10., 3.],
        [4, -1]
    ])

    Y_class = np.array([
        1.,
        -1.,
        -1.,
        -1.,
        -1.,
        1.
    ])

    Y_reg = np.array([
        2.5,
        -4.,
        -4.5,
        -9.,
        -8.,
        7.
    ])

    print("REGRESSION")
    model = create_linear_model(3)
    is_classif = False

    print(f"PREDICT BEFORE TRAIN: {predict_linear_model(model, np.array([0, 0]), is_classif)}")
    train_linear_model(
        model,
        X,
        Y_class if is_classif else Y_reg,
        0.001,
        5000,
        is_classif
    )
    print("TRAIN END")
    res = predict_linear_model(model, np.array([0, 0]), is_classif)
    print(f"PREDICT AFTER TRAIN: {res}")

    destroy_linear_model(model)

    print("CLASSIFICATION")
    model = create_linear_model(3)
    is_classif = True

    print(f"PREDICT BEFORE TRAIN: {predict_linear_model(model, np.array([0, 0]), is_classif)}")
    train_linear_model(
        model,
        X,
        Y_class if is_classif else Y_reg,
        0.001,
        5000,
        is_classif
    )
    print("TRAIN END")
    res = predict_linear_model(model, np.array([0, 0]), is_classif)
    print(f"PREDICT AFTER TRAIN: {res}")

    destroy_linear_model(model)

    print("END PYTHON")
