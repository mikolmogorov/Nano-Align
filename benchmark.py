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


def benchmark(mat_file, svr_file, write_output):
    events = sp.read_mat(mat_file)
    #sp.normalize(events)
    #clusters = sp.get_averages(events, TEST_AVG)
    clusters = sp.cluster_events(events)

    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)

    p_values = []
    for cluster in clusters:
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     nano_hmm.num_peaks)
        #discr_signal = (discr_signal - np.mean(discr_signal)) / np.std(discr_signal)

        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        p_values.append(p_value_raw)
        if write_output:
            print(len(cluster.events), p_value_raw)

        #nano_hmm.plot_raw_vs_theory(discr_signal, peptide)

    if write_output:
        print("Mean: ", np.mean(p_values))
        print("Median: ", np.median(p_values))
    return np.median(p_values)


TEST_AVG = 10


def main():
    benchmark(sys.argv[1], sys.argv[2], True)


if __name__ == "__main__":
    main()
