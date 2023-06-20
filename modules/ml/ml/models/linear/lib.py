import ctypes
from typing import Optional, List

# lib = ctypes.CDLL("your_rust_library.dll")  # Replace with the actual library filename
# lib = ctypes.CDLL("your_rust_library.so")  # Replace with the actual library filename

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/libml_core.so")


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
        x_train: List[int],
        lines: int,
        columns: int,
        y_train: List[int],
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

    return ml_lib.train_linear_model(
        model,
        x_train,
        lines,
        columns,
        y_train,
        y_train_columns,
        alpha,
        epochs,
        is_classification
    )


def predict_linear_model(model, sample_input, lines):
    lib.predict_linear_model.argtypes = [
        ctypes.POINTER(LinearRegressionModel),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32
    ]
    lib.predict_linear_model.restype = ctypes.c_double


def save_linear_model(model: int, filename: bytes):
    ml_lib.save_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p
    ]

    ml_lib.save_linear_model.restypes = None

    ml_lib.save_linear_model(model, filename)


def destroy_linear_model(model):
    lib.destroy_linear_model.argtypes = [ctypes.POINTER(LinearRegressionModel)]
    lib.destroy_linear_model.restypes = None


def load_linear_model(filename):
    lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.load_linear_model.restype = ctypes.POINTER(LinearRegressionModel)

    return lib.load_linear_model(filename)


# Load the Rust library using ctypes

if __name__ == '__main__':
    print("STARTING IN PYTHON")
    model = create_linear_model(4)
    print("VALUE: {}".format(model))
    print("CALLING SAVE MODEL")
    save_linear_model(
        model,
        b'lol.txt'
    )
    print("END PYTHON")
