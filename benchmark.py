#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt

from nanopore.nanohmm import NanoHMM, aa_to_weights
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

    for b in boxes:
        print(b)
    fig = plt.subplot()
    fig.set_yscale("log")
    fig.boxplot(boxes)
    plt.show()


def benchmark(mat_file, svr_file, write_output):
    events = sp.read_mat(mat_file)
    sp.normalize(events)
    clusters = sp.get_averages(events, 10)
    #clusters = sp.cluster_events(events)

    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)

    p_values = []
    for cluster in clusters:
        num_peaks = len(peptide) + 3
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     num_peaks)

        #likelihood, weights = nano_hmm.hmm(discr_signal)
        #p_value = nano_hmm.compute_pvalue(weights, peptide)

        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        p_values.append(p_value_raw)
        if write_output:
            print(len(cluster.events), p_value_raw)

    if write_output:
        print("Mean: ", np.mean(p_values))
        print("Median: ", np.median(p_values))
    return np.median(p_values)


def main():
    benchmark(sys.argv[1], sys.argv[2], True)
    #full_benchmark(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
