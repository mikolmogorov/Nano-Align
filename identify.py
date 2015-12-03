#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from Bio import SeqIO

import matplotlib.pyplot as plt
import numpy as np

from nanopore.nanohmm import NanoHMM, aa_to_weights
import nanopore.signal_proc as sp


def indetification_test(events, peptide, db_file, svr_file):
    nano_hmm = NanoHMM(len(peptide), svr_file)
    clusters = sp.get_averages(events, TRAIN_AVG, FLANK)
    #clusters = sp.cluster_events(events, FLANK)

    #build database
    database = {}
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = aa_to_weights(str(seq.seq))
    hist = defaultdict(int)

    #testing
    peptide_weights = aa_to_weights(peptide)
    scores = []
    pvals = []
    print("No\tSize\tProt_id\t\tMax_score\tTrue_score\tP-value")
    for num, cluster in enumerate(clusters):
        likelihood, weights = nano_hmm.hmm(cluster.consensus)
        p_value = nano_hmm.compute_pvalue(weights, peptide)
        true_score = nano_hmm.score(weights, peptide_weights)

        max_score = -sys.maxint
        chosen_prot = 0
        for prot_id, prot_seq in database.items():
            score = nano_hmm.score(prot_seq, weights)
            #nano_hmm.show_target_vs_decoy(weights, prot_seq, peptide)
            if score > max_score:
                max_score = score
                chosen_prot = prot_id

        hist[chosen_prot] += len(cluster.events)
        print("{0}\t{1}\t{2:10}\t{3:7.4f}\t\t{4:7.4f}\t\t{5}"
                .format(num, len(cluster.events), chosen_prot,
                        max_score, true_score, p_value))

    for prot_id in sorted(hist, reverse=True):
        print(prot_id, hist[prot_id])
    print("Correct: {0:5.2f}%"
                .format(100.0 * hist["target"] / sum(hist.values())))


TRAIN_AVG = 10
FLANK = 50

def main():
    if len(sys.argv) != 4:
        print("Usage: identify.py mat_file db_file svr_file")
        return 1

    events, peptide = sp.read_mat(sys.argv[1])
    indetification_test(events, peptide, sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
