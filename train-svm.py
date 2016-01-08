#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from string import maketrans
from itertools import product
import pickle
import os

import numpy as np
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import matplotlib

import nanopore.signal_proc as sp
import benchmark
import identify


AA_SIZE_TRANS = maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                          "MMMMMSSSSSSIIIIIIIIILLLL-")
def aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)


def _kmer_features(kmer):
    miniscule = kmer.count("M")
    small = kmer.count("S")
    intermediate = kmer.count("I")
    large = kmer.count("L")
    return (large, intermediate, small, miniscule)


def _peptide_to_features(peptide, window):
    num_peaks = len(peptide) + window - 1
    flanked_peptide = ("-" * (window - 1) + peptide +
                       "-" * (window - 1))
    features = []
    for i in xrange(0, num_peaks):
        kmer = flanked_peptide[i : i + window]
        feature = _kmer_features(kmer)

        features.append(feature)

    return features


def _get_features(clusters, window):
    features = []
    signals = []
    peptide = clusters[0].events[0].peptide
    num_peaks = len(peptide) + window - 1

    for cluster in clusters:
        discretized = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                    num_peaks)
        features.extend(_peptide_to_features(aa_to_weights(peptide), window))
        signals.extend(discretized)

    return features, signals


def _serialize_svr(svr, window, out_file):
    pickle.dump(svr, open(out_file, "wb"))


def _score_svr(svr, clusters, window):
    def rand_jitter(arr):
        stdev = .01*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    features = []
    signals = []
    peptide = clusters[0].events[0].peptide
    num_peaks = len(peptide) + window - 1

    for cluster in clusters:
        discretized = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                    num_peaks)
        features.extend(_peptide_to_features(aa_to_weights(peptide), window))
        signals.extend(discretized)

    print(svr.score(features, signals))
    #pca = PCA(2)
    #pca.fit(features)
    #new_x = pca.transform(features)
    #plt.hist(signals, bins=50)
    #plt.show()
    #plt.scatter(rand_jitter(new_x[:, 0]), rand_jitter(new_x[:, 1]), s=50,
    #            c=signals, alpha=0.5)
    #plt.show()


def _process_mats(mat_files):
    features = []
    signals = []
    for mat in mat_files:
        events = sp.read_mat(mat)
        sp.normalize(events)
        #train_events = sp.cluster_events(events)
        train_events = sp.get_averages(events, TRAIN_AVG)
        f, s = _get_features(train_events, WINDOW)
        features.extend(f)
        signals.extend(s)

    return features, signals


WINDOW = 4
TRAIN_AVG = 1


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


def just_train():
    if len(sys.argv) < 3:
        print("Usage: train-svm.py train_mat_1[,train_mat_2...] out_file")
        return 1

    train_features, train_signals = _process_mats(sys.argv[1:-1])
    #svr = SVR(kernel="rbf", gamma=0.01, epsilon=0.01, C=10)
    svr = SVR(kernel="rbf", C=1000, gamma=0.001, epsilon=0.01)
    svr.fit(train_features, train_signals)
    _serialize_svr(svr, WINDOW, sys.argv[-1])


def main():
    #cross_validate()
    just_train()


if __name__ == "__main__":
    main()
