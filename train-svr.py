#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Trains SVR model ans serializes it into a file
"""

from __future__ import print_function
import sys
import os
import argparse

import numpy as np

from nanoalign.__version__ import __version__
import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat
from nanoalign.pvalues_test import pvalues_test
from nanoalign.model_loader import store_model
from nanoalign.svr import SvrBlockade
from nanoalign.random_forest import RandomForestBlockade


def train_svr(mat_files, out_file, C=1000, gamma=0.001, epsilon=0.01):
    """
    Trains SVR with the given parameters
    """
    TRAIN_AVG = 1

    peptides = []
    signals = []
    for mat in mat_files:
        blockades = read_mat(mat)
        clusters = sp.preprocess_blockades(blockades, cluster_size=TRAIN_AVG,
                                           min_dwell=0.5, max_dwell=20)
        mat_peptide = clusters[0].blockades[0].peptide
        peptides.extend([mat_peptide] * len(clusters))

        for cluster in clusters:
            signals.append(sp.discretize(cluster.consensus, len(mat_peptide)))

    #model = SvrBlockade()
    #model.train(peptides, signals, C, gamma, epsilon)
    model = RandomForestBlockade()
    model.train(peptides, signals)
    store_model(model, out_file)


def cross_validate(train_mats, cv_mats, db_file, out_file):
    """
    Choosing the best parameters through cross-validation
    """
    CLUSTER_SIZE = 10

    eps_vec = [0.01, 0.001, 0.0001, 0.00001]
    C_vec = [1, 10, 100, 1000, 10000, 100000]
    gamma_vec = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    best_score = sys.maxint
    best_params = None

    print("C\tGam\tEps\tScore", file=sys.stderr)
    for C in C_vec:
        for gamma in gamma_vec:
            for eps in eps_vec:
                temp_file = os.path.basename(train_mats[0] + "_temp.pcl")
                train_svr(train_mats, temp_file, C, gamma, eps)

                scores = []
                for cv_mat in cv_mats:
                    pval, rank = pvalues_test(cv_mat, CLUSTER_SIZE, temp_file,
                                              db_file, False,
                                              open(os.devnull, "w"))
                    scores.append(rank)
                score = np.mean(scores)

                os.remove(temp_file)

                print("{0}\t{1}\t{2}\t{3}".format(C, gamma, eps, score),
                      file=sys.stderr)
                if score < best_score:
                    best_score = score
                    best_params = (C, gamma, eps)

    print(*best_params, file=sys.stderr)
    train_svr(train_mats, out_file, *best_params)


def main():
    parser = argparse.ArgumentParser(description="Nano-Align SVR model "
                                     "training", formatter_class= \
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("train_blockades", metavar="train_blockades",
                        help="comma-separated list of files with train "
                        "blockades (in mat format)")
    parser.add_argument("out_file", metavar="out_file",
                        help="path to the output SVR file "
                        "(in Python's pickle format)")
    parser.add_argument("-c", "--cv-blockades", dest="cv_blockades",
                        metavar="cv_blockades", help="comma-separated "
                        "list with blockades files for cross-valiadtion. "
                        "No CV, if not set", default=None)
    parser.add_argument("-d", "--cv-database", dest="cv_database",
                        metavar="cv_database", help="database file for CV "
                        "(in FASTA format). If not set, random database "
                        "is generated",
                        default=None)

    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    if args.cv_blockades is None:
        train_svr(args.train_blockades.split(","), args.out_file)
    else:
        cross_validate(args.train_blockades.split(","),
                       args.cv_blockades.split(","), args.cv_database,
                       args.out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
