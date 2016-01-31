#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import namedtuple
import math
import random

from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from nanopore.nanohmm import NanoHMM

import nanopore.signal_proc as sp


def find_peaks(signal):
    WINDOW = 6
    deriv = np.zeros(len(signal) - 2)
    for i in xrange(len(deriv)):
        deriv[i] = (signal[i + 2] - signal[i]) / 2

    peaks = []
    for pos in xrange(WINDOW / 2, len(deriv) - WINDOW / 2):
        left = deriv[pos - WINDOW / 2: pos]
        right = deriv[pos: pos + WINDOW / 2]

        if all(x > 0 for x in left) and all(y < 0 for y in right):
            peaks.append(pos)

    return peaks


def theoretical_signal(peptide, window_size):
    VOLUMES = {"I": 0.1688, "F": 0.2034, "V": 0.1417, "L": 0.1679,
               "W": 0.2376, "M": 0.1708, "A": 0.0915, "G": 0.0664,
               "C": 0.1056, "Y": 0.2036, "P": 0.1293, "T": 0.1221,
               "S": 0.0991, "H": 0.1673, "E": 0.1551, "N": 0.1352,
               "Q": 0.1611, "D": 0.1245, "K": 0.1713, "R": 0.2021}

    signal = []
    for i in xrange(-window_size + 1, len(peptide)):
        start, end = max(i, 0), min(i + window_size, len(peptide))
        volumes = np.array(map(VOLUMES.get, peptide[start:end]))
        value = sum(volumes) / window_size
        signal.append(value)
    return signal


def get_acids_positions(peptide, window_size, plot_len):
    num_peaks = len(peptide) + window_size - 1
    peak_shift = float(plot_len) / (num_peaks - 1)
    initial_shift = (window_size - 1) * peak_shift / 2
    positions = []
    for aa in xrange(len(peptide)):
        positions.append(initial_shift + aa * peak_shift)
    return positions


def compare_events(clusters, need_smooth):
    print("Comparing {0} clusters".format(len(clusters)))
    event_len = len(clusters[0].consensus)
    for cluster_1, cluster_2 in zip(clusters[:-1], clusters[1:]):
        event_1 = cluster_1.consensus
        event_2 = cluster_2.consensus
        prot = cluster_1.events[0].peptide

        if need_smooth:
            smooth_frac = float(1) / len(prot)
            event_1 = sp.smooth(event_1, smooth_frac)
            event_2 = sp.smooth(event_2, smooth_frac)

        #event_1 = (event_1 - np.mean(event_1)) / np.std(event_1)
        #event_2 = (event_2 - np.mean(event_2)) / np.std(event_2)

        event_1 = sp.trim_flank_noise(event_1)
        event_2 = sp.trim_flank_noise(event_2)
        event_1 = sp.discretize(event_1, len(prot) + 3)
        event_2 = sp.discretize(event_2, len(prot) + 3)

        plot_1 = event_1
        plot_2 = event_2

        print("Correlation", 1 - distance.correlation(plot_1, plot_2))
        plt.plot(np.repeat(plot_1, 2))
        plt.plot(np.repeat(plot_2, 2))
        plt.show()

        #plt.scatter(plot_1, plot_2)
        #plt.show()


def correlation(clusters):

    prot = clusters[0].events[0].peptide
    signals = map(lambda c: sp.discretize(sp.trim_flank_noise(c.consensus),
                                          len(prot) + 2), clusters)
    coeffs = []
    for event_1 in signals:
        for event_2 in signals:

            correlation = 1 - distance.correlation(event_1, event_2)
            coeffs.append(correlation)

    print(np.median(coeffs))


def plot_blockades(clusters, window, need_smooth):
    num_samples = len(clusters)
    prot = clusters[0].events[0].peptide

    model_volume = theoretical_signal(prot, window)
    #nanohmm = NanoHMM(len(prot), sys.argv[2])
    #model_svr = nanohmm.peptide_signal(prot)

    for cluster in clusters:
        event = cluster.consensus
        event = sp.trim_flank_noise(event)
        event_len = len(event)
        model_grid = [i * event_len / (len(model_volume) - 1)
                      for i in xrange(len(model_volume))]

        if need_smooth:
            smooth_frac = float(1) / len(prot)
            event = sp.smooth(event, smooth_frac)

        interp_fun = interp1d(model_grid, model_volume, kind="cubic")
        model_interp = interp_fun(xrange(event_len))

        ##
        #interp_fun = interp1d(model_grid, model_svr, kind="cubic")
        #svr_interp = interp_fun(xrange(event_len))
        #svr_scaled = (svr_interp - np.median(svr_interp)) / np.std(svr_interp)
        ##

        model_scaled = (model_interp - np.median(model_interp)) / np.std(model_interp)
        event = (event - np.median(event)) / np.std(event)
        ###

        event_plot = event
        model_plot = model_scaled

        #event_plot = sp.discretize(event_plot, len(prot))
        #model_plot = sp.discretize(model_plot, len(prot))

        print(1 - distance.correlation(event_plot, model_plot))

        #################
        ##pretty plotting
        x_axis = np.linspace(0, len(prot) + 1, len(event_plot))
        matplotlib.rcParams.update({"font.size": 16})
        fig = plt.subplot(111)
        fig.plot(x_axis, event_plot, label="Empirical signal", linewidth=1.5)
        fig.plot(x_axis, model_plot, label="MV model", linewidth=1.5)
        #fig.plot(x_axis, svr_scaled, label="SVR model", linewidth=1.5)

        fig.spines["right"].set_visible(False)
        fig.spines["top"].set_visible(False)
        fig.get_xaxis().tick_bottom()
        fig.get_yaxis().tick_left()
        fig.set_xlim(0, len(prot) + 1)
        fig.set_xlabel("Putative AA position")
        fig.set_ylabel("Normalized signal")

        legend = fig.legend(loc="lower left", frameon=False)
        for label in legend.get_lines():
            label.set_linewidth(2)
        for label in legend.get_texts():
            label.set_fontsize(16)

        #adding AAs text:
        #event_mean = np.mean(event)
        #acids_pos = get_acids_positions(prot, window, len(event_plot))
        #for i, aa in enumerate(prot):
        #    plt.text(acids_pos[i], event_mean-2, aa, fontsize=16)

        plt.show()
        ################


WINDOW = 4
AVERAGE = 1
SMOOTH = False


def main():
    if len(sys.argv) < 3:
        print("Usage: plot.py mat_file")
        return 1

    events = sp.read_mat(sys.argv[1])
    sp.normalize(events)
    #clusters = sp.cluster_events(events)
    clusters = sp.get_averages(events, AVERAGE)

    plot_blockades(clusters, WINDOW, SMOOTH)
    #correlation(clusters)
    #compare_events(clusters, SMOOTH)


if __name__ == "__main__":
    main()
