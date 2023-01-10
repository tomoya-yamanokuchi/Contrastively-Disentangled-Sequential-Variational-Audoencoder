import numpy as np


def negative_log_gaussian_density_expectation(var):
    dim = var.shape[-1]
    c1  = dim * np.log(2*np.pi)
    c2  = dim
    return 0.5 * (c1 + np.log(var).sum(-1) + c2)


def entropy_Hy(var_y, eps=1E-16):
    H_y = negative_log_gaussian_density_expectation(var_y)
    return H_y.sum(-1) # sum over timestep


def entropy_Hyx(var_yx):
    H_yxi = negative_log_gaussian_density_expectation(var_yx)
    H_yxi = H_yxi.sum(axis=-1) # sum over timestep
    return H_yxi.mean(axis=0)  # expectation over samples


def inception_score(var_yx, var_y):
    inner_exp = 0.5 * (np.log(var_y).sum(-1) - np.log(var_yx).sum(-1)).mean(axis=0)
    is_score  = np.exp(inner_exp)
    return is_score.sum() # sum over timestep



