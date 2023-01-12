import numpy as np


def negative_log_gaussian_density_expectation(var):
    dim = var.shape[-1]
    c1  = dim * np.log(2*np.pi)
    c2  = dim
    return 0.5 * (c1 + np.log(var).sum(-1) + c2)


# def entropy_Hy(var_yx):
#     H_y = negative_log_gaussian_density_expectation(var_yx)
#     # import ipdb; ipdb.set_trace()
#     return H_y.sum(-1) # sum over timestep


def entropy_Hyx(var_yx):
    H_yxi = negative_log_gaussian_density_expectation(var_yx)
    H_yxi = H_yxi.sum(axis=-1) # sum over timestep
    return H_yxi.mean(axis=0)  # expectation over samples


def inception_score(var_yx, log_density_py):
    H_yxi    = negative_log_gaussian_density_expectation(var_yx).sum(-1) # sum over timestep
    is_score = np.exp( (-H_yxi).mean(axis=0) - log_density_py)
    return is_score


def kl_divergence(q, p):
    '''
    - gaussian KL-divergence KL(q||p)
    - variance is diagonal
    - shape = (num_batch, step, dim)
    '''
    q_mean, q_logvar = q
    p_mean, p_logvar = p
    dim = q_mean.shape[-1]

    logdet_q = q_logvar.sum(axis=-1)
    logdet_p = p_logvar.sum(axis=-1)

    q_var = np.exp(q_logvar)
    p_var = np.exp(p_logvar)
    trace = (q_var / p_var).sum(axis=-1)

    y = p_mean - q_mean
    squared_mahalanobis_distance = (y**2 / p_var).sum(axis=-1)

    kld = 0.5 * (logdet_p - logdet_q - dim + trace + squared_mahalanobis_distance)
    return kld.sum(-1).mean()