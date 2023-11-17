import numpy as np


def bisquare_kernel(distance, bandwidth):
    weights = (1 - (distance / bandwidth) ** 2) ** 2
    return np.where(distance <= bandwidth, weights, 0)
