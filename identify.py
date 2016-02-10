#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from Bio import SeqIO

import matplotlib.pyplot as plt
import numpy as np
import argparse

from nanoalign.svr_blockade import SvrBlockade
from nanoalign.identifier import Identifier
from nanoalign.blockade import read_mat
import nanoalign.signal_proc as sp


#TODO: cluster detalization, full identify


def make_database(db_file, peptide):
    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)
        if database[seq.id] == peptide:
            target_id = seq.id

    return database, target_id


def identification_test(blockades_file, cluster_size, svr_file, db_file=None):
    """
    Performs protein identification and report results
    """
    RANDOM_DB_SIZE = 10000
    blockade_model = SvrBlockade()
    blockade_model.load_from_pickle(svr_file)
    identifier = Identifier(blockade_model)

    blockades = read_mat(blockades_file)

    true_peptide = blockades[0].peptide
    if db_file is None:
        identifier.random_database(true_peptide, RANDOM_DB_SIZE)
        target_id = "target"
        db_len = RANDOM_DB_SIZE
    else:
        database, target_id = make_database(db_file, true_peptide)
        identifier.set_database(database)
        db_len = len(database)

    clusters = sp.preprocess_blockades(blockades, cluster_size=cluster_size,
                                       min_dwell=0.5, max_dwell=20)

    print("\nNo\tSize\tBest_id\t\tBest_dst\tTrg_dst\t\tTrg_rank\t"
          "Trg_pval", file=sys.stderr)
    p_values = []
    ranks = []
    for num, cluster in enumerate(clusters):
        db_ranking = identifier.rank_db_proteins(cluster.consensus)

        target_rank = None
        target_dist = None
        for rank, (prot_id, prot_dist) in enumerate(db_ranking):
            if prot_id == target_id:
                target_rank = rank
                target_dist = prot_dist
        p_value = float(target_rank) / db_len

        p_values.append(p_value)
        ranks.append(target_rank)

        print("{0}\t{1}\t{2:10}\t{3:5.2f}\t\t{4:5.2f}\t\t{5}\t\t{6:6.4}"
               .format(num + 1, len(cluster.blockades), db_ranking[0][0],
                       db_ranking[0][1], target_dist, target_rank + 1, p_value),
              file=sys.stderr)

    print("\nMedian p-value: {0:7.4f}".format(np.median(p_values)),
          file=sys.stderr)
    print("Median target rank: {0:d}".format(int(np.median(ranks))),
          file=sys.stderr)

"""
def identify(events_file, db_file, svr_file, write_output):
    events = sp.read_mat(events_file)
    events = sp.filter_by_time(events, 0.5, 20)
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
"""


"""
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
"""

"""
def full_identify(mat_file, db_file, svr_file):
    events = sp.read_mat(mat_file)
    events = sp.filter_by_time(events, 0.5, 20)
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

    for b in boxes:
        print(",".join(map(str, b)))
"""


def main():
    parser = argparse.ArgumentParser(description="Nano-Align protein "
                                     "identification", formatter_class= \
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("blockades_file", metavar="blockades_file",
                        help="path to blockades file (in mat format)")
    parser.add_argument("svr_file", metavar="svr_file",
                        help="path to SVR file (in Python's pickle format)")
    parser.add_argument("-c", "--cluster-size", dest="cluster_size", type=int,
                        default=10, help="blockades cluster size")
    parser.add_argument("-d", "--database", dest="database",
                        metavar="database", help="database file (in FASTA format). "
                        "If not set, random database is generated", default=None)
    parser.add_argument("--version", action="version", version="0.1b")
    args = parser.parse_args()

    identification_test(args.blockades_file, args.cluster_size,
                        args.svr_file, args.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())
