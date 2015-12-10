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

    correct_weights = aa_to_weights(peptide)
    print(correct_weights, "\n")
    profile = [[] for x in xrange(len(correct_weights))]

    identified = 0
    print("Size\tSequence\tHMM_score\tFrac_corr\tFit_pvalue")
    for cluster in clusters:
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     nano_hmm.num_peaks)
        score, weights = nano_hmm.hmm(discr_signal)
        p_value = nano_hmm.compute_pvalue(weights, peptide)
        p_value_raw = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        if p_value < 0.01:
            identified += 1

        accuracy = _hamming_dist(weights, correct_weights)
        accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
        print("{0}\t{1}\t{2:5.2f}\t{3:5.2f}\t{4}\t{5}"
                    .format(len(cluster.events), weights, score, accuracy,
                            p_value, p_value_raw))
        for pos, aa in enumerate(weights):
            profile[pos].append(aa)

        nano_hmm.show_fit(discr_signal, weights, peptide)

    profile = "".join(map(_most_common, profile))
    accuracy = _hamming_dist(profile, correct_weights)
    accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
    print()
    print(profile, accuracy)
    print("Identified:", float(identified) / len(clusters))


TRAIN_AVG = 1
TEST_AVG = 5


def main():
    events = sp.read_mat(sys.argv[1])
    sp.normalize(events)
    clusters = sp.get_averages(events, TEST_AVG)
    #clusters = sp.cluster_events(events)
    benchmarks(clusters, sys.argv[2])


if __name__ == "__main__":
    main()
