import numpy as np


def row_standardize(matrix):
    return (matrix.T / matrix.sum(axis=1)).T


def center_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n


def ones_vector(n):
    return np.ones((n, 1))
