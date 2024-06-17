from scipy.signal import cheby1, sosfilt
import numpy as np


def bandpass(x: np.ndarray, axis: int, low_freq: float, high_freq: float, sample_freq: float) -> np.ndarray:
    """
    Apply bandpass filtering and down sampling.

    :param x: input data
    :param axis: signal axis
    :param low_freq: low frequency
    :param high_freq: high frequency
    :param sample_freq: sampling rate
    :return: filtered data
    """
    sos = cheby1(8, 1, (low_freq, high_freq), btype='bandpass', analog=False, output='sos', fs=sample_freq)
    return sosfilt(sos, x, axis=axis)


def z_score(x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply z-score standardization.

    :param x: input data
    :param axis: signal axis
    :return: standardized data, mean, standard deviation
    """
    x_mean = np.mean(x, axis=axis, keepdims=True)
    x_std = np.std(x, axis=axis, keepdims=True)
    return (x - x_mean) / x_std, x_mean, x_std


__all__ = ['bandpass', 'z_score']
