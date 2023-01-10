from torch import Tensor

def to_numpy(x: Tensor):
    return x.to('cpu').numpy()