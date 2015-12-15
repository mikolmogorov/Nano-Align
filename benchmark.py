#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np

from nanopore.nanohmm import NanoHMM, aa_to_weights
import nanopore.signal_proc as sp


def _hamming_dist(str_1, str_2):
    res = 0
    for a, b in zip(str_1, str_2):
        res += int(a != b)
    return res


def _most_common(lst):
    return max(set(lst), key=lst.count)


def benchmarks(clusters, svr_file):
    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)

    p_values = []
    for cluster in clusters:
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     nano_hmm.num_peaks)
        #discr_signal = (discr_signal - np.mean(discr_signal)) / np.std(discr_signal)
        discr_signal -= np.mean(discr_signal)

        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        p_values.append(p_value_raw)
        print(len(cluster.events), p_value_raw)

        #nano_hmm.plot_raw_vs_theory(discr_signal, peptide)

    print("Mean: ", np.mean(p_values))
    print("Median: ", np.median(p_values))


TRAIN_AVG = 1
TEST_AVG = 10


def main():
    events = sp.read_mat(sys.argv[1])
    sp.normalize(events)
    clusters = sp.get_averages(events, TEST_AVG)
    #clusters = sp.cluster_events(events)
    benchmarks(clusters, sys.argv[2])


if __name__ == "__main__":
    main()
