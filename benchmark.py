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


def benchmarks(events, peptide, svr_file):
    nano_hmm = NanoHMM(len(peptide), svr_file)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    clusters = sp.get_averages(events, TEST_AVG, FLANK)
    #clusters = sp.cluster_events(events, FLANK)

    correct_weights = aa_to_weights(peptide)
    print(correct_weights, "\n")
    profile = [[] for x in xrange(len(correct_weights))]

    identified = 0
    print("Size\tSequence\tHMM_score\tFrac_corr\tFit_pvalue")
    for cluster in clusters:
        score, weights = nano_hmm.hmm(cluster.consensus)
        p_value = nano_hmm.compute_pvalue(weights, peptide)
        p_value_raw = nano_hmm.compute_pvalue_raw(cluster.consensus, peptide)
        if p_value < 0.01:
            identified += 1

        accuracy = _hamming_dist(weights, correct_weights)
        accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
        print("{0}\t{1}\t{2:5.2f}\t{3:5.2f}\t{4}\t{5}"
                    .format(len(cluster.events), weights, score, accuracy,
                            p_value, p_value_raw))
        for pos, aa in enumerate(weights):
            profile[pos].append(aa)

        discr_signal = sp.discretize(sp.normalize(cluster.consensus,
                                                  nano_hmm.num_peaks),
                                     nano_hmm.num_peaks)
        nano_hmm.show_fit(discr_signal, weights, peptide)

    profile = "".join(map(_most_common, profile))
    accuracy = _hamming_dist(profile, correct_weights)
    accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
    print()
    print(profile, accuracy)
    print("Identified:", float(identified) / len(clusters))


TRAIN_AVG = 1
TEST_AVG = 10
FLANK = 50


def main():
    events, peptide = sp.read_mat(sys.argv[1])
    benchmarks(events, peptide, sys.argv[2])


if __name__ == "__main__":
    main()
