import numpy as np


def itr(p: np.ndarray) -> np.ndarray:
    """
    Calculate ITR (deprecated, please use stable_itr() instead)

    :param p: accuracy of character predictions
    :return: ITR
    """
    return (60 * (p * np.log2(p) + (1 - p) * np.log2((1 - p) / 35) + np.log2(36))) / (2.5 + 2.1 * np.arange(1, 16, 1))


def stable_itr(p: np.ndarray) -> np.ndarray:
    """
    Calculate ITR (numerical stable)

    :param p: accuracy of character predictions
    :return: ITR
    """
    conditions = [np.isclose(p, 0, atol=1e-6), np.isclose(p, 1, atol=1e-6)]
    functions = [
        lambda x: np.log2(1 / 35),
        lambda x: 0,
        lambda x: x * np.log2(x) + (1 - x) * np.log2((1 - x) / 35),
    ]
    return (60 * (np.piecewise(p, conditions, functions) + np.log2(36))) / (2.5 + 2.1 * np.arange(1, 16, 1))


def mean_itr(acc):
    return sum(stable_itr(acc)) / 15


__all__ = ['stable_itr', 'itr', 'mean_itr']
