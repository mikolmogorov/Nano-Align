#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress

from nanopore.nanohmm import NanoHMM
import nanopore.signal_proc as sp


def _hamming_dist(str_1, str_2):
    res = 0
    for a, b in zip(str_1, str_2):
        res += int(a != b)
    return res


def _most_common(lst):
    return max(set(lst), key=lst.count)


def full_benchmark(mat_file, svr_file):
    events = sp.read_mat(mat_file)
    events = sp.filter_by_time(events, 0.5, 20)
    sp.normalize(events)
    peptide = events[0].peptide
    num_peaks = len(peptide) + 3
    nano_hmm = NanoHMM(len(peptide), svr_file)

    boxes = []
    for avg in xrange(1, 21):
        p_values = []
        for _ in xrange(avg):
            clusters = sp.get_averages(events, avg)
            for cluster in clusters:
                discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                             num_peaks)
                p_value = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
                p_values.append(p_value)

        boxes.append(p_values)
        print(avg, np.median(p_values))

    for box in boxes:
        print(",".join(map(str, box)))


def fancy_plot(errors):
    VOLUMES = {"I": 169, "F": 203, "V": 142, "L": 168,
               "W": 238, "M": 171, "A": 92, "G": 66,
               "C": 106, "Y": 204, "P": 129, "T": 122,
               "S": 99, "H": 167, "E": 155, "N": 135,
               "Q": 161, "D": 124, "K": 171, "R": 202}
    HYDRO =   {"I": 100, "F": 92, "V": 79, "L": 100,
               "W": 84, "M": 74, "A": 47, "G": 0,
               "C": 52, "Y": 49, "P": -46, "T": 13,
               "S": -7, "H": -42, "E": 8, "N": -41,
               "Q": -18, "D": -18, "K": -37, "R": -26}

    sorted_aa = sorted(errors.keys(), key=HYDRO.get)
    sorted_values = map(errors.get, sorted_aa)

    x_axis = range(1, len(sorted_aa) + 1)
    medians = map(lambda x: np.median(x), sorted_values)
    poly = np.polyfit(x_axis, medians, 1)
    poly_fun = np.poly1d(poly)

    matplotlib.rcParams.update({"font.size": 16})
    fig = plt.subplot()
    bp = fig.boxplot(sorted_values, sym="", notch=True)
    fig.set_xticklabels(sorted_aa)

    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.get_xaxis().tick_bottom()
    fig.get_yaxis().tick_left()
    fig.set_ylim(-2, 2)

    for y in [-1, 0, 1]:
        fig.plot([y] * 20, "--",
                 lw=0.5, color="black", alpha=0.3)

    for box in bp["boxes"]:
        box.set(color="#7570b3", linewidth=2)
    for whisker in bp["whiskers"]:
        whisker.set(color="white", linewidth=2)
    for cap in bp["caps"]:
        cap.set(color="white", linewidth=2)
    for median in bp["medians"]:
        median.set(color="red", linewidth=2)

    fig.plot(x_axis, poly_fun(x_axis), "r-", linewidth=1.5)

    fig.set_xlabel("Amino acids (sorted by hydrophillicity)")
    fig.set_ylabel("Signed error")
    plt.show()


def benchmark(mat_file, svr_file, write_output):
    events = sp.read_mat(mat_file)
    events = sp.filter_by_time(events, 0.5, 20)
    sp.normalize(events)
    clusters = sp.get_averages(events, 10)
    #clusters = sp.cluster_events(events)

    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)

    p_values = []
    errors = defaultdict(list)

    for cluster in clusters:
        num_peaks = len(peptide) + 3
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     num_peaks)

        for aa, e in nano_hmm.get_errors(peptide, discr_signal):
            errors[aa].append(e)

        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        p_values.append(p_value_raw)
        if write_output:
            print(len(cluster.events), p_value_raw)

    fancy_plot(errors)

    if write_output:
        print("Mean: ", np.mean(p_values))
        print("Median: ", np.median(p_values))
    return np.median(p_values)


def main():
    benchmark(sys.argv[1], sys.argv[2], True)
    #full_benchmark(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
