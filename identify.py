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


def rank_db_proteins(nano_hmm, signal, database):
    max_score = -sys.maxint
    #chosen_prot = 0
    scores = {}
    for prot_id, prot_seq in database.items():
        score = nano_hmm.signal_peptide_score(signal, prot_seq)
        scores[prot_id] = score

    return sorted(scores.items(), key=lambda i: i[1], reverse=True)


def detalize_cluster(nano_hmm, cluster, database):
    norm_events = sp.get_averages(cluster.events, 1)
    global_rankings = defaultdict(list)
    for enum, event in enumerate(norm_events):
        discr_signal = sp.discretize(sp.trim_flank_noise(event.consensus),
                                     nano_hmm.num_peaks)
        rankings = rank_db_proteins(nano_hmm, discr_signal, database)
        for i in xrange(len(rankings)):
            global_rankings[rankings[i][0]].append(i)
            if rankings[i][0] == "target":
                target_rank = i
        print("\tSignal {0}, target rank = {1}".format(enum, target_rank))

    for prot in global_rankings:
        global_rankings[prot] = np.mean(global_rankings[prot])
    global_rankings = sorted(global_rankings.items(), key=lambda i: i[1])

    print("\tRanking")
    for prot, rank in global_rankings[:10]:
        print("\t\t{0}\t{1}".format(prot, rank))
        #for prot, prot_score in rankings[:10]:
        #    print("\t\t{0}\t{1}".format(prot, prot_score))


def indetification_test(events, db_file, svr_file):
    #sp.normalize(events)
    #clusters = sp.get_averages(events, TRAIN_AVG)
    clusters = sp.cluster_events(events)
    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)

    #build database
    database = {}
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)
    hist = defaultdict(int)

    #testing
    peptide_weights = peptide
    scores = []
    pvals = []
    print("No\tSize\tProt_id\t\tMax_score\tTrue_score\tP-value\tP-value_trg")
    for num, cluster in enumerate(clusters):
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     nano_hmm.num_peaks)

        chosen_prot, max_score = rank_db_proteins(nano_hmm, discr_signal,
                                                  database)[0]
        true_score = nano_hmm.signal_peptide_score(discr_signal, peptide)

        p_value = nano_hmm.compute_pvalue_raw(discr_signal, database[chosen_prot])
        p_value_target = nano_hmm.compute_pvalue_raw(discr_signal, peptide)

        hist[chosen_prot] += len(cluster.events)
        print("{0}\t{1}\t{2:10}\t{3:7.4f}\t\t{4:7.4f}\t\t{5}\t\t{6}"
                .format(num, len(cluster.events), chosen_prot,
                        max_score, true_score, p_value, p_value_target))

        #detalize_cluster(nano_hmm, cluster, database)

    for prot_id in sorted(hist, reverse=True):
        print(prot_id, hist[prot_id])
    print("Correct: {0:5.2f}%"
                .format(100.0 * hist["target"] / sum(hist.values())))


TRAIN_AVG = 10

def main():
    if len(sys.argv) != 4:
        print("Usage: identify.py mat_file db_file svr_file")
        return 1

    events = sp.read_mat(sys.argv[1])
    indetification_test(events, sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
