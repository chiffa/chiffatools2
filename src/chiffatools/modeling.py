__author__ = 'andrei'

import numpy as np
from linalg_routines import rm_nans
from scipy.stats import t


def logistic_regression(TF, T0, concentrations, background_std):

    def get_1p_bounds(mean, std, dof):
        return t.interval(0.99, dof, mean, std)

    mask = concentrations == 0.0
    vals_at_0 = rm_nans(TF[:, mask, :])
    max_capacity = get_1p_bounds(np.mean(vals_at_0),
                                 np.sqrt(np.var(vals_at_0) + background_std**2),
                                 vals_at_0.shape[0])[1]*1.05

    compensation_T0 = -np.log2(max_capacity/T0-1)[:, :, np.newaxis]
    compensation_TF = -np.log2(max_capacity/TF-1)

    alphas = compensation_TF - compensation_T0

    return alphas
