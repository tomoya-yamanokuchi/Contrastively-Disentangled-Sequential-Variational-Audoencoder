import numpy as np
from torch import Tensor
from copy import deepcopy

def save_as_numpy_scalar(tensor_dict: dict, path):
    numpy_dict = {}
    for key, val in tensor_dict.items():
        key = key.split("/")[-1] # to ignore a group for tensorbord
        assert isinstance(val, Tensor)
        x = val.to("cpu").detach().numpy()
        numpy_dict[key] = deepcopy(x)
    np.save(file=path + "/numpy_summary", arr=numpy_dict)