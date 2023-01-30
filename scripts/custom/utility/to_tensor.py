import torch

def to_tensor(x):
    return torch.from_numpy(x).contiguous().type(torch.FloatTensor)