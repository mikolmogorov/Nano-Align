#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Runs identificatino test and report p-values
"""

from __future__ import print_function
import sys
from collections import defaultdict
import argparse

from Bio import SeqIO
import numpy as np

from nanoalign.svr_blockade import SvrBlockade
from nanoalign.identifier import Identifier
from nanoalign.blockade import read_mat
import nanoalign.signal_proc as sp


def _make_database(db_file, peptide):
    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)
        if database[seq.id] == peptide:
            target_id = seq.id

    return database, target_id


def identification_test(blockades_file, cluster_size, svr_file, db_file,
                        single_blockades=False):
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
        database, target_id = _make_database(db_file, true_peptide)
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
        if single_blockades:
            _detalize_cluster(identifier, cluster, db_ranking[0][0], target_id)

    print("\nMedian p-value: {0:7.4f}".format(np.median(p_values)),
          file=sys.stderr)
    print("Median target rank: {0:d}".format(int(np.median(ranks))),
          file=sys.stderr)


def _detalize_cluster(identifier, cluster, top_id, target_id):
    """
    Prints information about each single blockade inside cluster
    """
    single_blockades = sp.preprocess_blockades(cluster.blockades,
                                               cluster_size=1)
    global_rankings = defaultdict(list)
    for num, cluster in enumerate(single_blockades):
        rankings = identifier.rank_db_proteins(cluster.consensus)
        for i in xrange(len(rankings)):
            global_rankings[rankings[i][0]].append(i)
            if rankings[i][0] == target_id:
                target_rank = i
            if rankings[i][0] == top_id:
                winner_rank = i
        print("\tSignal {0}, target = {1}, consensus top = {2}"
                .format(num, target_rank, winner_rank))

    for prot in global_rankings:
        global_rankings[prot] = np.mean(global_rankings[prot])
    global_rankings = sorted(global_rankings.items(), key=lambda i: i[1])

    print("\tRanking:")
    for prot, rank in global_rankings[:10]:
        print("\t\t{0}\t{1}".format(prot, rank))



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
                        metavar="database", help="database file (in FASTA "
                        "format). If not set, random database is generated",
                        default=None)
    parser.add_argument("-s", "--single-blockades", action="store_true",
                        default=False, dest="single_blockades",
                        help="print statistics for each blockade in a cluster")

    parser.add_argument("--version", action="version", version="0.1b")
    args = parser.parse_args()

    identification_test(args.blockades_file, args.cluster_size,
                        args.svr_file, args.database, args.single_blockades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
