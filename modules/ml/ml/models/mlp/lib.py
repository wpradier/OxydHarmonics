import ctypes
import numpy as np
from matplotlib import pyplot as plt

ml_lib = ctypes.CDLL("../../../ml_core/target/debug/libml_core.so")


class MultilayerPerceptron:
    def __init__(self, pointer: int, structure: np.array):
        self.pointer = pointer
        self.structure = structure


class MLPPrediction:
    def __init__(self, pointer: int, value: np.array):
        self.pointer = pointer
        self.value = value

    def __str__(self):
        return str(self.value)

    def __del__(self):
        ml_lib.destroy_mlp_prediction.argtypes = [ctypes.c_void_p]
        ml_lib.destroy_mlp_prediction.restype = None

        ml_lib.destroy_mlp_prediction(self.pointer)

def create_mlp_model(structure: np.array) -> MultilayerPerceptron:
    ml_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    ml_lib.create_mlp_model.restype = ctypes.c_void_p

    c_structure = structure.astype(int).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    pointer = ml_lib.create_mlp_model(c_structure, len(structure))
    return MultilayerPerceptron(pointer, structure)


def train_mlp_model(
        model: MultilayerPerceptron,
        x_train: np.array,
        y_train: np.array,
        alpha: float,
        epochs: int,
        is_classification: bool
):
    c_x_train = x_train.astype(float).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_y_train = y_train.astype(float).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ml_lib.train_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_bool
    ]
    ml_lib.train_mlp_model.restype = None

    return ml_lib.train_mlp_model(
        model.pointer,
        c_x_train,
        len(x_train),
        len(x_train[0]),
        c_y_train,
        len(y_train),
        len(y_train[0]),
        alpha,
        epochs,
        is_classification
    )


def predict_mlp_model(model: MultilayerPerceptron, sample_input: np.array, is_classification: bool) -> np.array:
    ml_lib.predict_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_bool
    ]
    ml_lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double * model.structure[-1])

    c_sample = sample_input.astype(float).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    prediction = ml_lib.predict_mlp_model(
            model.pointer,
            c_sample,
            len(sample_input),
            is_classification
        )

    return np.ctypeslib.as_array(prediction, (model.structure[-1],)).flatten()[:model.structure[-1]]


def destroy_mlp_model(model: MultilayerPerceptron):
    ml_lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
    ml_lib.destroy_mlp_model.restypes = None

    return ml_lib.destroy_mlp_model(model.pointer)


if __name__ == '__main__':
    print("TESTING MLP")
    model = create_mlp_model(np.array([2, 1]))
    print("CREATED MODEL")
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        [1],
        [-1],
        [-1]
    ])

    print("AMATATRAIN AAAAAAAAAAAAAH")
    train_mlp_model(
        model,
        X,
        Y,
        0.01,
        1000,
        True
    )

    print("TRAINING DONE")

    res = predict_mlp_model(model, X[0], True)

    print("GOT PREDICTION RESULT")
    print(f"PREDICTION: {res}")

    destroy_mlp_model(model)

    print("TESTING MULTICLASS")

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [-1, -1, -1] for p in X])

    X = X[[not np.all(arr == [-1, -1, -1]) for arr in Y]]
    Y = Y[[not np.all(arr == [-1, -1, -1]) for arr in Y]]

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    plt.clf()

    mlp_model = create_mlp_model(np.array([2, 3]))

    train_mlp_model(
        mlp_model,
        X,
        Y,
        0.001,
        1000,
        True
    )

    prediction = predict_mlp_model(mlp_model, np.array([-0.5, -0.5]), True)

    print(f"PREDICTION IN PYTHON: {prediction}")

    destroy_mlp_model(mlp_model)