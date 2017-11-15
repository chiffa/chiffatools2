"""
A set of tools that come handy to perform basic statistics tasks
"""

__author__ = 'andrei'

import numpy as np
from linalg_routines import rm_nans
from scipy.stats import ttest_ind
from itertools import combinations
import scipy.cluster.hierarchy as sch
from scipy.stats import norm, t, poisson


def t_test_matrix(samples, jacknife_size=50):
    """
    Performs series of t_tests between the segments of current lane divided by breakpoints. In addition to
    that, uses an inner re-sampling to prevent discrepancies due to lane size

    Alternative: use normalized differences in level (mean/std)
    Even better: use Tuckey test

    :param current_lane: fused list of chromosomes
    :param breakpoints: list of positions where HMM detected significant differences between elements
    :param jacknife_size: if None or 0, will run T-test on the entire sample. Otherwise will use the provided integert
                            to know how many elements from each collection to sample for a t-test.
    :return: matrix of P values student's t_test of difference between average ploidy segments
    """

    def inner_prepare(array):
        if jacknife_size:
            return rm_nans(np.random.choice(array,
                                            size=(jacknife_size,),
                                            replace=True))
        else:
            return rm_nans(array)

    samples_number = len(samples)
    p_vals_matrix = np.empty((samples_number, samples_number))
    p_vals_matrix.fill(np.NaN)
    for i, j in combinations(range(0, samples_number), 2):
        _, p_val = ttest_ind(inner_prepare(samples[i]), inner_prepare(samples[j]), False)
        p_vals_matrix[i, j] = p_val
    return p_vals_matrix


def t_test_collapse(set_of_samples):
    """
    Takes in a set of samples and returns per-sample means and the groups of means that are statistically not different

    :param set_of_samples:
    :return:
    """
    print 'set of samples length:', len(set_of_samples)

    nanmeans = [np.nanmean(x) for x in set_of_samples]

    t_mat = t_test_matrix(set_of_samples, None)  # generate T-test matrix

    t_mat[np.isnan(t_mat)] = 0
    t_mat = t_mat + t_mat.T
    np.fill_diagonal(t_mat, 1)
    ct_mat = t_mat.copy()
    ct_mat[t_mat < 0.01] = 0.01

    y_matrix = sch.linkage(ct_mat, method='centroid')
    clust_alloc = sch.fcluster(y_matrix, 0.95, criterion='distance') - 1  # merge on the 5% rule

    # groups together elements that are not statistcally significantly different at the 5% level
    accumulator = [[] for _ in range(0, max(clust_alloc) + 1)]
    for loc, item in enumerate(nanmeans):
        accumulator[clust_alloc[loc]].append(item)

    accumulator = np.array([np.nanmean(np.array(_list)) for _list in accumulator])

    collapsed_means = np.empty_like(nanmeans)
    collapsed_means.fill(np.nan)

    for i, j in enumerate(clust_alloc.tolist()):
        collapsed_means[i] = accumulator[j]

    return nanmeans, collapsed_means


def tukey_outliers(set_of_means, FDR=0.005, supporting_interval=0.5, verbose=False):
    """
    Performs Tukey quintile test for outliers from a normal distribution with defined false discovery rate

    :param set_of_means:
    :param FDR:
    :return:
    """
    # false discovery rate v.s. expected falses v.s. power
    q1_q3 = norm.interval(supporting_interval)
    FDR_q1_q3 = norm.interval(1 - FDR)
    multiplier = (FDR_q1_q3[1] - q1_q3[1]) / (q1_q3[1] - q1_q3[0])
    l_means = len(set_of_means)

    q1 = np.percentile(set_of_means, 50*(1-supporting_interval))
    q3 = np.percentile(set_of_means, 50*(1+supporting_interval))
    high_fence = q3 + multiplier*(q3 - q1)
    low_fence = q1 - multiplier*(q3 - q1)

    if verbose:
        print 'FDR:', FDR
        print 'q1_q3', q1_q3
        print 'FDRq1_q3', FDR_q1_q3
        print 'q1, q3', q1, q3
        print 'fences', high_fence, low_fence

    if verbose:
        print "FDR: %s %%, expected outliers: %s, outlier 5%% confidence interval: %s"% (FDR*100, FDR*l_means,
                                                                                  poisson.interval(0.95, FDR*l_means))

    ho = (set_of_means < low_fence).nonzero()[0]
    lo = (set_of_means > high_fence).nonzero()[0]

    return lo, ho


def get_outliers(lane, FDR):
    """
    Gets the outliers in a lane with a given FDR and sets all non-outliers in the lane to NaNs

    :param lane:
    :param FDR:
    :return:
    """
    lo, ho = tukey_outliers(lane, FDR)
    outliers = np.empty_like(lane)
    outliers.fill(np.nan)
    outliers[ho] = lane[ho]
    outliers[lo] = lane[lo]

    return outliers


def p_stabilize(array, percentile):
    p_low = np.percentile(rm_nans(array), percentile)
    p_high = np.percentile(rm_nans(array), 100-percentile)
    array[array < p_low] = p_low
    array[array > p_high] = p_high
    return array


def get_t_distro_outlier_bound_estimation(array, background_std):
    narray = rm_nans(array)

    low, up = t.interval(0.95, narray.shape[0]-1, np.mean(narray), np.sqrt(np.var(narray)+background_std**2))
    up, low = (up-np.mean(narray), np.mean(narray)-low)

    return max(up, low)


def quantile_normalization(ref, target):
    sorted_ref = np.sort(ref)
    argsort_target = np.argsort(target)
    rev_argsort = np.argsort(argsort_target)
    normalized_target = sorted_ref[rev_argsort]

    return normalized_target


def zero_preserving_quantile_normalization(ref, target, zero=1):
    zero = np.ones_like(ref)*zero
    non_zero_in_both = np.logical_and(ref != zero, target != zero)

    non_zero_ref = ref[non_zero_in_both]
    non_zero_target = target[non_zero_in_both]

    sorted_nz_ref = np.sort(non_zero_ref)
    argsort_nz_target = np.argsort(non_zero_target)
    rev_argsort = np.argsort(argsort_nz_target)
    normalized_nz_target = sorted_nz_ref[rev_argsort]

    normalized_target = np.zeros_like(target)
    normalized_target[non_zero_in_both] = normalized_nz_target
    normalized_target[np.logical_not(non_zero_in_both)] = target[np.logical_not(non_zero_in_both)]

    # TODO: add correction of elements set to 0 to sampled from the reference-generated distribution

    return normalized_target


def conditional_cut_off(static_error, relative_error, multiplier=2):

    def total_error_function(x):
        x = np.power(10, x)

        variance_with_bounds = np.sqrt(np.power(relative_error*x, 2)+np.power(static_error, 2))*multiplier*2

        _min = (-variance_with_bounds+np.sqrt(variance_with_bounds**2+4*x**2))/2  # quadratic solve
        _max = _min + variance_with_bounds

        worst_fraction = np.log2(_max/_min)

        return worst_fraction

    total_error_function = np.vectorize(total_error_function)

    return total_error_function
