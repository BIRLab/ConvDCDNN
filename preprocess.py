from scipy.signal import cheby1, sosfilt, decimate
import numpy as np


def bandpass_and_decimate(x, axis, low_freq, high_freq, sample_freq, down_freq=None):
    sos = cheby1(8, 1, (low_freq, high_freq), btype='bandpass', analog=False, output='sos', fs=sample_freq)
    x = sosfilt(sos, x, axis=axis)
    if down_freq is None:
        return x
    return decimate(x, round(sample_freq / down_freq), axis=axis)


def z_score(x, axis):
    x_mean = np.mean(x, axis=axis, keepdims=True)
    x_std = np.std(x, axis=axis, keepdims=True)
    return (x - x_mean) / x_std, x_mean, x_std


def scale(x, axis):
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    return (x - min_val) / (max_val - min_val)


__all__ = ['bandpass_and_decimate', 'z_score', 'scale']
