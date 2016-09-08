#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
This module estimates fluctuation frequency of blockades,
which is associated with a protein length and other features
"""

from __future__ import print_function
import sys
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.fftpack as fftpack
from scipy import signal

from nanoalign.__version__ import __version__
import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat


def savitsky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Savitsky-Golay smoothing
    (taken from http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay)
    """
    try:
       window_size = np.abs(np.int(window_size))
       order = np.abs(np.int(order))
    except ValueError, msg:
       raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
       raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
       raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def gcd_fuzz(numbers):
    """
    Computes "approximate" common divisor
    """
    vals = []
    for div in xrange(1, 300):
        rems = []
        for num in numbers:
            rem = min(num % div, (num // div + 1) * div - num)
            rems.append((float(rem)))

        vals.append(np.mean(rems) / div)

    gcd_x, gcd_y = sp.find_peaks(vals, minimum=True, ranged=True)
    gcds = filter(lambda x: 40 < x < 500, gcd_x)

    return vals, gcds


def frequency_distribution(blockades_file, detailed):
    """
    Plots the frequency distribution
    """
    blockades = read_mat(blockades_file)
    blockades = sp._fractional_blockades(blockades)
    blockades = sp._filter_by_duration(blockades, 0.5, 20)

    peaks_count = {}
    for blockade in blockades:
        if detailed:
            detailed_plots(blockade)

        signal = blockade.eventTrace[1000:-1000]
        xx, yy = sp.find_peaks(signal)
        peaks_count[blockade] = len(xx) / blockade.ms_Dwell * 5 / 4

    mean = np.mean(peaks_count.values())
    errors = map(lambda e: peaks_count[e] - mean, blockades)
    lengths = map(lambda e: e.ms_Dwell, blockades)

    f, (s1, s2) = plt.subplots(2)
    s1.scatter(lengths, errors)
    s2.hist(peaks_count.values(), bins=100)
    plt.show()


def detailed_plots(blockade):
    """
    Plots extra information about a single blockade
    """
    #FFT
    fft = fftpack.rfft(blockade.eventTrace)

    #Getting mean distance between peaks
    xx, yy = sp.find_peaks(blockade.eventTrace)
    diff_1 = sorted(np.array(xx)[1:] - np.array(xx)[:-1])
    diff_2 = sorted(np.array(xx)[2:] - np.array(xx)[:-2])
    diff_3 = sorted(np.array(xx)[3:] - np.array(xx)[:-3])
    diff = diff_1 + diff_2 + diff_3
    diff = np.array(diff) * blockade.ms_Dwell
    d_hist, bin_edges = np.histogram(diff, bins=100, range=(100, 1000))

    window = signal.gaussian(21, std=1)
    smooth_diff = np.convolve(d_hist, window, mode="same")
    smooth_diff = savitsky_golay(smooth_diff, 5, 3)

    #Guessing gcds of peak distance distributino
    hist_x, hist_y = sp.find_peaks(smooth_diff)
    char_peaks = map(lambda x: int(bin_edges[x + 1]), hist_x)
    gcd_plot, gcds = gcd_fuzz(char_peaks)

    f, (s1, s2, s3, s4) = plt.subplots(4)
    s1.plot(blockade.eventTrace)
    s1.scatter(xx, yy)
    s1.set_xlim(0, 10000)
    s1.set_ylabel("Blockade")

    s2.plot(np.linspace(0, 1000 / blockade.ms_Dwell, 1000),
            savitsky_golay(fft ** 2, 31, 3)[:1000])
    s2.set_yscale("log")
    s2.set_ylabel("FFT")

    s3.plot(bin_edges[1:], smooth_diff)
    s3.set_ylabel("Peak dist. distr.")

    s4.plot(gcd_plot)
    s4.set_ylabel("Peak dist. gcd")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Estimates blockades "
                                     "frequency", formatter_class= \
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("blockades_file", metavar="blockades_file",
                        help="path to blockades file (in mat format)")
    parser.add_argument("-d", "--detailed", action="store_true",
                        default=False, dest="detailed",
                        help="detailed plots for each blockade")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    frequency_distribution(args.blockades_file, args.detailed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
