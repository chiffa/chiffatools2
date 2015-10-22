"""
Rolling frame methods
"""
__author__ = 'andrei'


import numpy as np


def rolling_window(base_array, window_size):
    """
    Extracts a rolling subarray from the current array of a provided size

    :param base_array: array to which we want to apply the rolling window
    :param window_size: the size of the rolling window
    :return:
    """
    shape = base_array.shape[:-1] + (base_array.shape[-1] - window_size + 1, window_size)
    strides = base_array.strides + (base_array.strides[-1],)
    return np.lib.stride_tricks.as_strided(base_array, shape=shape, strides=strides)


def rolling_mean(base_array, window_size):
    """

    :param base_array:
    :param window_size:
    :return:
    """
    rar = rolling_window(base_array, window_size)
    return np.pad(np.nanmean(rar, 1), (window_size/2, (window_size-1)/2), mode='edge')


def rolling_std(base_array, window_size, quintiles=False):
    """

    :param base_array:
    :param window_size:
    :param quintiles:
    :return:
    """
    rar = rolling_window(base_array, window_size)
