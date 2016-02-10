#!/usr/bin/env python

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Trains SVR model ans serializes it into a file
"""

from __future__ import print_function
import sys
import os

import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat
from nanoalign.svr_blockade import SvrBlockade


def simple_train(mat_files, out_file):
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

    svr_model = SvrBlockade()
    svr_model.train(peptides, signals)
    svr_model.store_pickle(out_file)


"""
def cross_validate():
    if len(sys.argv) != 5:
        print("Usage: train-svm.py train_signals cv_signals db_file out_file")
        return 1

    train_mats = sys.argv[1].split(",")
    cv_mats = sys.argv[2].split(",")
    db_file = sys.argv[3]

    train_features, train_signals = _process_mats(train_mats)
    #cv_features, cv_signals = _process_mats(cv_mats)

    #eps_vec = [0.0001, 0.001, 0.01, 0.1]
    eps_vec = [0.01, 0.001, 0.0001, 0.00001]
    C_vec = [1, 10, 100, 1000, 10000, 100000]
    gamma_vec = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    best_score = sys.maxint
    best_svr = None
    best_params = None

    print("C\tGam\tEps\tScore")
    for C in C_vec:
        for gamma in gamma_vec:
            for eps in eps_vec:
                svr = SVR(kernel="rbf", gamma=gamma, epsilon=eps, C=C)
                svr.fit(train_features, train_signals)

                temp_file = os.path.basename(train_mats[0] + "_temp.pcl")
                _serialize_svr(svr, WINDOW, temp_file)
                scores = []
                for cv_mat in cv_mats:
                    #scores.append(benchmark.benchmark(cv_mat, temp_file, False))
                    scores.append(identify.identify(cv_mat, db_file,
                                                   temp_file, False))
                os.remove(temp_file)
                score = np.mean(scores)

                print("{0}\t{1}\t{2}\t{3}".format(C, gamma, eps, score))
                if score < best_score:
                    best_score = score
                    best_svr = svr
                    best_params = (C, gamma, eps)

    print(*best_params)
    _serialize_svr(best_svr, WINDOW, sys.argv[4])
"""

#TODO: parser, cross-validation
def main():
    if len(sys.argv) < 3:
        print("Usage: train-svm.py train_mat_1[,train_mat_2...] out_file",
              file=sys.stderr)
        return 1

    mat_files = sys.argv[1:-1]
    out_file = sys.argv[-1]
    simple_train(mat_files, out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
