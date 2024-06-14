from scipy.signal import cheby1, sosfilt, decimate
import numpy as np


def bandpass_and_decimate(x: np.ndarray, axis: int, low_freq: float, high_freq: float, sample_freq: float, down_freq: float | None = None) -> np.ndarray:
    """
    Apply bandpass filtering and down sampling.

    :param x: input data
    :param axis: signal axis
    :param low_freq: low frequency
    :param high_freq: high frequency
    :param sample_freq: sampling rate
    :param down_freq: down sampling frequency
    :return: filtered data
    """
    sos = cheby1(8, 1, (low_freq, high_freq), btype='bandpass', analog=False, output='sos', fs=sample_freq)
    x = sosfilt(sos, x, axis=axis)
    if down_freq is None:
        return x
    return decimate(x, round(sample_freq / down_freq), axis=axis)


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


def scale(x, axis):
    """
    Scale signal.

    :param x: input data
    :param axis: signal axis
    :return: scaled data
    """
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    return (x - min_val) / (max_val - min_val)


__all__ = ['bandpass_and_decimate', 'z_score', 'scale']
