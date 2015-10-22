__author__ = 'ank'

import matplotlib.pyplot as plt
from time import time


def rs(matrix, name):
    plt.title(name)
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.show()


def debug_wrapper(funct):

    def check_matrix(*args, **kwargs):
        result = funct(*args, **kwargs)
        if not isinstance(result, tuple):
            rs(result, funct.__name__)
        else:
            rs(result[0], funct.__name__)
        check_matrix.__name__ = funct.__name__
        check_matrix.__doc__ = funct.__doc__
        return result

    return check_matrix


def time_wrapper(funct):

    def time_execution(*args, **kwargs):
        start = time()
        result = funct(*args, **kwargs)
        print funct.__name__, time() - start
        time_execution.__doc__ = funct.__doc__
        return result

    return time_execution
