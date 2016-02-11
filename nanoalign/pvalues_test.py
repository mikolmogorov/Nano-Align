#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Performs identification test and report p-values
"""

from collections import defaultdict

from Bio import SeqIO
import numpy as np

from nanoalign.svr_blockade import SvrBlockade
from nanoalign.identifier import Identifier
from nanoalign.blockade import read_mat
import nanoalign.signal_proc as sp


def _make_database(db_file, peptide):
    """
    Reads protein database
    """
    database = {}
    target_id = None
    for seq in SeqIO.parse(db_file, "fasta"):
        database[seq.id] = str(seq.seq)
        if database[seq.id] == peptide:
            target_id = seq.id

    return database, target_id


def pvalues_test(blockades_file, cluster_size, svr_file, db_file,
                 single_blockades, ostream):
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

    ostream.write("\nNo\tSize\tBest_id\t\tBest_dst\tTrg_dst\t\tTrg_rank\t"
                     "Trg_pval\n")
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

        ostream.write("{0}\t{1}\t{2:10}\t{3:5.2f}\t\t{4:5.2f}\t\t{5}\t\t{6:6.4}\n"
               .format(num + 1, len(cluster.blockades), db_ranking[0][0],
                       db_ranking[0][1], target_dist, target_rank + 1, p_value))
        if single_blockades:
            _detalize_cluster(identifier, cluster, db_ranking[0][0],
                              target_id, ostream)

    ostream.write("\nMedian p-value: {0:7.4f}\n".format(np.median(p_values)))
    ostream.write("Median target rank: {0:d}\n".format(int(np.median(ranks))))

    return np.median(p_values), int(np.median(ranks))


def _detalize_cluster(identifier, cluster, top_id, target_id, ostream):
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
        ostream.write("\tSignal {0}, target = {1}, consensus top = {2}\n"
                        .format(num, target_rank, winner_rank))

    for prot in global_rankings:
        global_rankings[prot] = np.mean(global_rankings[prot])
    global_rankings = sorted(global_rankings.items(), key=lambda i: i[1])

    ostream.write("\tRanking:\n")
    for prot, rank in global_rankings[:10]:
        ostream.write("\t\t{0}\t{1}\n".format(prot, rank))
