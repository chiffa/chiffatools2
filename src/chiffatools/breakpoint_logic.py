__author__ = 'andrei'

import numpy as np
from linalg_routines import rm_nans

def brp_retriever(array, breakpoints_set):
    """
    Retrieves values on the segments defined by the breakpoints.

    :param array:
    :param breakpoints_set:
    :return: the values in array contained between breakpoints
    """
    breakpoints_set = sorted(list(set(breakpoints_set))) # sorts tbe breakpoints
    if breakpoints_set[-1] == array.shape[0]:
        breakpoints_set = breakpoints_set[:-1]  # just in case end of array was already included
    values = np.split(array, breakpoints_set) # retrieves the values
    return values


def brp_setter(breakpoints_set, prebreakpoint_values):
    """
    Creates an array of the size defined by the biggest element of the breakpoints set and sets the
    intervals values to prebreakpoint_values. It assumes that the largest element of breakpoints set
    is equal to the size of desired array

    :param array:
    :param breakpoints_set:
    :param prebreakpoint_values:
    :return:
    """
    breakpoints_set = sorted(list(set(breakpoints_set))) # sorts the breakpoints
    assert(len(breakpoints_set) == len(prebreakpoint_values))
    support = np.empty((breakpoints_set[-1], )) # creates array to be filled
    support.fill(np.nan)

    pre_brp = 0 # fills the array
    for value, brp in zip(prebreakpoint_values, breakpoints_set):
        support[pre_brp:brp] = value
        pre_brp = brp

    return support


def pull_breakpoints(contingency_list):
    """
    A method to extract breakpoints separating np.array regions with the same value.

    :param contingency_list: np.array containing regions of identical values
    :return: list of breakpoint indexes
    """
    no_nans_parsed = rm_nans(contingency_list)
    contingency = np.lib.pad(no_nans_parsed[:-1] == no_nans_parsed[1:], (1, 0), 'constant', constant_values=(True, True))
    nans_contingency = np.zeros(contingency_list.shape).astype(np.bool)
    nans_contingency[np.logical_not(np.isnan(contingency_list))] = contingency
    breakpoints = np.nonzero(np.logical_not(nans_contingency))[0].tolist()
    return breakpoints


def generate_breakpoint_mask(breakpoints):
    """
    generates mask assigning a different integer to each breakpoint

    :param breakpoints:
    :return:
    """
    support = np.zeros((np.max(breakpoints), ))
    pre_brp = 0
    for i, brp in enumerate(breakpoints):
        support[pre_brp:brp] = i
        pre_brp = brp
    return support