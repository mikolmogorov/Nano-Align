#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
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

        for v, e in nano_hmm.get_errors(peptide, discr_signal):
            errors[int(v * 1000)].append(e)

        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        p_values.append(p_value_raw)
        if write_output:
            print(len(cluster.events), p_value_raw)

    """
    medians = map(lambda x: np.median(x), errors.values())
    poly = np.polyfit(errors.keys(), medians, 1)
    poly_fun = np.poly1d(poly)

    fig = plt.subplot()
    fig.boxplot(errors.values(), positions=errors.keys(), sym="", widths=3)
    fig.plot(errors.keys(), poly_fun(errors.keys()), "r-")
    fig.set_xticks(np.linspace(60, 210, 11))
    #fig.set_yticks(np.linspace(0, 2, 5))
    fig.set_xlabel("AA volume, A^3")
    fig.set_ylabel("Signed error")
    plt.show()
    """

    if write_output:
        print("Mean: ", np.mean(p_values))
        print("Median: ", np.median(p_values))
    return np.median(p_values)


def main():
    #benchmark(sys.argv[1], sys.argv[2], True)
    full_benchmark(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
