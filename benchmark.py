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


def benchmarks(events, peptide):
    nano_hmm = NanoHMM(peptide)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)
    #test_events = sp.cluster_events(events)

    nano_hmm.learn_emissions_distr(train_events)
    nano_hmm.score_svm(test_events)
    correct_weights = aa_to_weights(nano_hmm.peptide)
    print(correct_weights, "\n")
    profile = [[] for x in xrange(len(correct_weights))]

    identified = 0
    identified_raw = 0
    print("Sequence\t\tHMM_score\tFrac_corr\tFit_pvalue\tRaw_pvalue")
    for event in test_events:
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
        score, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        p_value_raw = nano_hmm.compute_pvalue_raw(event)
        if p_value < 0.01:
            identified += 1
        if p_value_raw < 0.01:
            identified_raw += 1

        accuracy = _hamming_dist(weights, correct_weights)
        accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
        print("{0}\t{1:5.2f}\t{2}\t{3}\t{4}".format(weights, score, accuracy,
                                                    p_value, p_value_raw))
        for pos, aa in enumerate(weights):
            profile[pos].append(aa)

        nano_hmm.show_fit(event, weights)

    profile = "".join(map(_most_common, profile))
    accuracy = _hamming_dist(profile, correct_weights)
    accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
    print()
    print(profile, accuracy)
    print("Identified:", float(identified) / len(test_events))
    print("Identified raw:", float(identified_raw) / len(test_events))


TRAIN_AVG = 1
TEST_AVG = 5
FLANK = 50


def main():
    events, peptide = sp.read_mat(sys.argv[1])
    benchmarks(events, peptide)


if __name__ == "__main__":
    main()
