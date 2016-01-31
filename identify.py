#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from Bio import SeqIO

import matplotlib.pyplot as plt
import numpy as np

from nanopore.nanohmm import NanoHMM
import nanopore.signal_proc as sp


def rank_db_proteins(nano_hmm, signal, database):
    max_score = -sys.maxint
    scores = {}
    discretized = {}

    for prot_id, prot_seq in database.items():
        num_peaks = len(prot_seq) + 3
        if len(prot_seq) not in discretized:
            d = sp.discretize(sp.trim_flank_noise(signal), num_peaks)
            discretized[len(prot_seq)] = d

        disc_signal = discretized[len(prot_seq)]
        score = nano_hmm.signal_peptide_score(disc_signal, prot_seq)
        scores[prot_id] = score

    return sorted(scores.items(), key=lambda i: i[1], reverse=True)


def detalize_cluster(nano_hmm, cluster, database, winner, target_id):
    norm_events = sp.get_averages(cluster.events, 1)
    global_rankings = defaultdict(list)
    for enum, event in enumerate(norm_events):
        rankings = rank_db_proteins(nano_hmm, event.consensus, database)
        for i in xrange(len(rankings)):
            global_rankings[rankings[i][0]].append(i)
            if rankings[i][0] == target_id:
                target_rank = i
            if rankings[i][0] == winner:
                winner_rank = i
        print("\tSignal {0}, target = {1}, winner = {2}".format(enum, target_rank,
                                                                winner_rank))

    for prot in global_rankings:
        global_rankings[prot] = np.mean(global_rankings[prot])
    global_rankings = sorted(global_rankings.items(), key=lambda i: i[1])

    print("\tRanking")
    for prot, rank in global_rankings[:10]:
        print("\t\t{0}\t{1}".format(prot, rank))
        #for prot, prot_score in rankings[:10]:
        #    print("\t\t{0}\t{1}".format(prot, prot_score))


def identify(events_file, db_file, svr_file, write_output):
    events = sp.read_mat(events_file)
    sp.normalize(events)
    clusters = sp.get_averages(events, 5)
    #clusters = sp.cluster_events(events)

    peptide = clusters[0].events[0].peptide
    num_peaks = len(peptide) + 3
    nano_hmm = NanoHMM(len(peptide), svr_file)

    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)
        if database[seq.id] == peptide:
            target_id = seq.id
    hist = defaultdict(int)

    scores = []
    pvals = []
    target_ranks = []
    if write_output:
        print("No\tSize\tProt_id\t\tMax_score\tTrue_score\tP-value\t"
              "P-value_trg\tTrg_rank")

    for num, cluster in enumerate(clusters):
        true_discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                          num_peaks)
        rankings = rank_db_proteins(nano_hmm, cluster.consensus, database)
        chosen_prot, max_score = rankings[0]
        target_rank = None
        for i, prot in enumerate(rankings):
            if prot[0] == target_id:
                target_rank = i
                break
        target_ranks.append(target_rank)

        true_score = nano_hmm.signal_peptide_score(true_discr_signal, peptide)

        p_value = nano_hmm.compute_pvalue_raw(true_discr_signal, database[chosen_prot])
        p_value_target = nano_hmm.compute_pvalue_raw(true_discr_signal, peptide)

        hist[chosen_prot] += len(cluster.events)
        if write_output:
            print("{0}\t{1}\t{2:10}\t{3:5.2f}\t{4:5.2f}\t{5}\t{6}\t{7}"
                    .format(num, len(cluster.events), chosen_prot,
                            max_score, true_score, p_value,
                            p_value_target, target_rank))

        #detalize_cluster(nano_hmm, cluster, database, chosen_prot, target_id)

    if write_output:
        for prot_id in sorted(hist, reverse=True):
            print(prot_id, hist[prot_id])
        print("Correct: {0:5.2f}%"
                    .format(100.0 * hist[target_id] / sum(hist.values())))

    return np.mean(target_ranks)


def full_identify(mat_file, db_file, svr_file):
    events = sp.read_mat(mat_file)
    sp.normalize(events)
    peptide = events[0].peptide
    num_peaks = len(peptide) + 3
    nano_hmm = NanoHMM(len(peptide), svr_file)

    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)[:len(peptide)]
        if database[seq.id] == peptide:
            target_id = seq.id

    boxes = []
    for avg in xrange(1, 21):
        p_values = []
        for _ in xrange(avg):
            clusters = sp.get_averages(events, avg)
            for cluster in clusters:
                discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                             num_peaks)
                rankings = rank_db_proteins(nano_hmm, cluster.consensus, database)
                target_rank = None
                for i, prot in enumerate(rankings):
                    if prot[0] == target_id:
                        target_rank = i
                        break

                #p_value = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
                p_value = float(target_rank) / len(database)
                p_values.append(p_value)

        boxes.append(p_values)
        print(avg, np.median(p_values))

    #for b in boxes:
    #    print(b)
    fig = plt.subplot()
    fig.set_yscale("log")
    fig.boxplot(boxes)
    fig.set_xlabel("Consensus size")
    fig.set_ylabel("P-value")
    plt.show()


def main():
    if len(sys.argv) != 4:
        print("Usage: identify.py mat_file db_file svr_file")
        return 1

    #identify(sys.argv[1], sys.argv[2], sys.argv[3], True)
    full_identify(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
