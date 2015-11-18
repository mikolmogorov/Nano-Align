#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from Bio import SeqIO

from nanopore.nanohmm import NanoHMM, aa_to_weights
import nanopore.signal_proc as sp


def indetification_test(events, peptide, db_file):
    nano_hmm = NanoHMM(peptide)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)
    #test_events = sp.cluster_events(events, FLANK)
    nano_hmm.learn_emissions_distr(train_events)
    nano_hmm.score_svm(test_events)

    #build database
    database = {}
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = aa_to_weights(str(seq.seq))
    hist = defaultdict(int)

    #testing
    misspredicted = 0
    peptide_weights = aa_to_weights(peptide)
    print("Event\tProt_id\t\tMax_score\tTrue_score\tP-value")
    for num, event in enumerate(test_events):
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
        likelihood, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        true_score = nano_hmm.score(weights, peptide_weights)

        max_score = -sys.maxint
        chosen_prot = 0
        for prot_id, prot_seq in database.items():
            score = nano_hmm.score(prot_seq, weights)
            if score > max_score:
                max_score = score
                chosen_prot = prot_id

        if chosen_prot != "target":
            misspredicted += 1
        hist[chosen_prot] += 1
        print("{0}\t{1:10}\t{2:7.4f}\t\t{3:7.4f}\t\t{4:7.2e}"
                .format(num, chosen_prot, max_score, true_score, p_value))

    for prot_id in sorted(hist, reverse=True):
        print(prot_id, hist[prot_id])
    print("Misspredicted:", float(misspredicted) / len(test_events))


TRAIN_AVG = 1
TEST_AVG = 5
FLANK = 50

def main():
    if len(sys.argv) != 3:
        print("Usage: identify.py mat_file db_file")
        return 1

    events, peptide = sp.read_mat(sys.argv[1])
    indetification_test(events, peptide, sys.argv[2])


if __name__ == "__main__":
    main()
