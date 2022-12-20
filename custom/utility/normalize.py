def normalize(x, x_min, x_max, m, M):
    a = (x - x_min) / (x_max - x_min)
    return a * (M - m) + m