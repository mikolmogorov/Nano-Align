#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

from nanopore.nanohmm import NanoHMM, aa_to_weights
import nanopore.signal_proc as sp


def indetification_test(events, peptide):
    nano_hmm = NanoHMM(peptide)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)
    #test_events = sp.cluster_events(events)
    nano_hmm.learn_emissions_distr(train_events)
    nano_hmm.score_svm(test_events)

    #build database
    database = []
    peptide_weights = aa_to_weights(nano_hmm.peptide)
    weights_list = list(peptide_weights)
    for _ in xrange(DB_SIZE):
        database.append("".join(weights_list))
        random.shuffle(weights_list)
    hist = defaultdict(int)

    #testing
    misspredicted = 0
    print("Event\tProt_id\tMax_score\tTrue_score\tP-value")
    for num, event in enumerate(test_events):
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
        likelihood, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        true_score = nano_hmm.score(weights, peptide_weights)

        max_score = -sys.maxint
        chosen_prot = 0
        for prot_id, db_prot in enumerate(database):
            score = nano_hmm.score(db_prot, weights)
            if score > max_score:
                max_score = score
                chosen_prot = prot_id

        if chosen_prot != 0:
            misspredicted += 1
        hist[chosen_prot] += 1
        print("{0}\t{1}\t{2:7.4f}\t{3:7.4f}\t{4:7.2e}"
                .format(num, chosen_prot, max_score, true_score, p_value))

    for prot_id, freq in hist.items():
        print(prot_id, freq)
    print("Misspredicted:", float(misspredicted) / len(test_events))


TRAIN_AVG = 1
TEST_AVG = 5
FLANK = 50
DB_SIZE = 1000


def main():
    events, peptide = sp.read_mat(sys.argv[1])
    indetification_test(events, peptide)


if __name__ == "__main__":
    main()
