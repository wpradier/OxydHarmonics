import ctypes
import numpy as np

#Need to set the correct path
my_lib = ctypes.CDLL("./../target/debug/libml.dylib")


npl = np.array([2, 3, 1])
npl_size = len(npl)

my_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
my_lib.create_mlp_model.restype = ctypes.c_void_p

model_pointer = my_lib.create_mlp_model(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), npl_size)