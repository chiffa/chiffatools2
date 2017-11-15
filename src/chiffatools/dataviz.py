"""
All credits for the implementation and suggestions go to sega_sai (stackoverflow):
http://stackoverflow.com/questions/10439961/efficiently-create-a-density-plot-for-high-density-regions-points-for-sparse-re
"""
__author__ = 'ank'


import matplotlib.pyplot as plt
import numpy as np
from scipy import histogram2d
from scipy.stats import gaussian_kde
from stats import zero_preserving_quantile_normalization, conditional_cut_off


def better2D_desisty_plot(xdat, ydat, thresh=3, bins=(100, 100)):
    xyrange = [[min(xdat), max(xdat)], [min(ydat), max(ydat)]]
    distortion = (xyrange[1][1] - xyrange[1][0]) / \
        (xyrange[0][1] - xyrange[0][0])
    xdat = xdat * distortion

    xyrange = [[min(xdat), max(xdat)], [min(ydat), max(ydat)]]
    hh, locx, locy = histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    # values of the histogram where the points are
    hhsub = hh[posx[ind] - 1, posy[ind] - 1]
    xdat1 = xdat[ind][hhsub < thresh]  # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan  # fill the areas with low density by NaNs

    plt.imshow(
        np.flipud(
            hh.T),
        cmap='jet',
        extent=np.array(xyrange).flatten(),
        interpolation='none')
    plt.plot(xdat1, ydat1, '.')


def violin_plot(ax, data, pos, bp=False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos) - min(pos)
    w = min(0.15 * max(dist, 1.0), 0.5)
    for d, p in zip(data, pos):
        k = gaussian_kde(d)  # calculates the kernel density
        m = k.dataset.min()  # lower bound of violin
        M = k.dataset.max()  # upper bound of violin
        x = np.arange(m, M, (M - m) / 100.)  # support for violin
        v = k.evaluate(x)  # violin profile (density curve)
        v = v / v.max() * w  # scaling the violin to the available space
        ax.fill_betweenx(x, p, v + p, facecolor='y', alpha=0.3)
        ax.fill_betweenx(x, p, -v + p, facecolor='y', alpha=0.3)
    if bp:
        ax.boxplot(data, notch=1, positions=pos, vert=1)


def kde_compute(bi_array, nbins=30, samples=10, show=True):

    overload = bi_array.shape[1] / float(samples)

    # In fact we are willing to evaluate what is the probability of encoutering at least ONE element in case of a generation;
    # Each random pull generates not one single point, butlots of them

    x, y = bi_array

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data
    # extents
    k = gaussian_kde(bi_array)
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = np.tanh(k(np.vstack([xi.flatten(), yi.flatten()])) * overload)

    if show:
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))

    return lambda x: np.tanh(k(x) * overload)


def smooth_histogram(data):
    fltr = np.logical_not(np.isnan(data))
    density = gaussian_kde(data[fltr].flatten())
    xs = np.linspace(data[fltr].min(), data[fltr].max(), 200)
    plt.plot(xs, density(xs), 'k')


def ma_plot(value_set1, value_set2, set1_lbl, set2_lbl,
            point_names=None,
            quantile_normalize=True,
            # dbscan_cut=False
            ):

    if quantile_normalize:
        value_set2 = zero_preserving_quantile_normalization(value_set1, value_set2)

    y_vals = np.log2(value_set2/value_set1)
    x_vals = np.log10(value_set2*value_set1)/2
    plt.plot(x_vals, y_vals, 'ko')

    cut_function = conditional_cut_off(10e6, 1.1)

    filter_lower = np.logical_and(y_vals < -3.32, y_vals < -cut_function(x_vals))
    filter_higher = np.logical_and(y_vals > 3.32, y_vals > cut_function(x_vals))

    sig_selector = np.logical_or(filter_lower, filter_higher)
    plt.plot(x_vals[sig_selector], y_vals[sig_selector], 'ro')
    lspace = np.linspace(x_vals.min(), x_vals.max())
    plt.plot(lspace, cut_function(lspace), 'r:')
    plt.plot(lspace, -cut_function(lspace), 'r:')

    for label, x, y in zip(point_names[sig_selector], x_vals[sig_selector], y_vals[sig_selector]):
        plt.annotate(label, xy=(x,y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            # bbox=dict(boxstyle='round, pad=0.0', fc='gray', alpha=0.0),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

    sig_under = point_names[filter_lower]
    sig_over = point_names[filter_higher]

    strict_under = point_names[np.logical_and(filter_lower, value_set2 < 10)]
    strict_over = point_names[np.logical_and(filter_higher, value_set1 < 10)]

    plt.ylabel(r"$log_2(\frac{%s}{%s})$" % (set2_lbl, set1_lbl))
    plt.xlabel(r"$log_{10}(\sqrt{%s*%s})$" % (set2_lbl, set1_lbl))

    return sig_under, sig_over, strict_under, strict_over

if __name__ == "__main__":
    from numpy.random import normal
    N = 1e5
    xdat, ydat = np.random.normal(size=N), np.random.normal(1, 0.6, size=N)
    better2D_desisty_plot(xdat, ydat)
    plt.show()

    pos = range(5)
    data = [normal(size=100) for i in pos]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    violin_plot(ax, data, pos, bp=1)
    plt.show()

    np.random.seed(1977)

    # Generate 200 correlated x,y points
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    kde_compute(data.T, nbins=20)
    plt.show()

