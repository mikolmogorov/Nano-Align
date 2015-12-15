#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random
from string import maketrans
from itertools import product
import pickle

import numpy as np
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import matplotlib

import nanopore.signal_proc as sp


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

    #return svr
    return features, signals


def _serialize_svr(svr, window, out_file):
    pickle.dump(svr, open(out_file, "wb"))
    #all_states = product("-MSIL", repeat=window)
    #all_states = sorted(map("".join, all_states))

    #with open(out_file, "w") as f:
    #    for state in all_states:
    #        feature = np.array(_kmer_features(state)).reshape(1, -1)
    #        f.write("{0}\t{1}\n".format(state, svr.predict(feature)[0]))


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


WINDOW = 4
TRAIN_AVG = 1
FLANK = 1


def main():
    if len(sys.argv) < 3:
        print("Usage: train-svm.py mat_file_1[,mat_file_2...] out_file")
        return 1

    mat_files = sys.argv[1:-1]
    features = []
    signals = []
    for mat in mat_files:
        events = sp.read_mat(mat)
        sp.normalize(events)
        #clusters = sp.cluster_events(events, FLANK)
        train_events = sp.get_averages(events, TRAIN_AVG)
        f, s = _get_features(train_events, WINDOW)
        features.extend(f)
        signals.extend(s)

    svr = SVR(kernel="rbf", gamma=10, epsilon=0.001, C=0.01)
    #svr = RANSACRegressor()
    svr.fit(features, signals)
    #print(svr.estimator_.coef_)

    #for mat in mat_files:
    #    events = sp.read_mat(mat)
    #    test_events = sp.get_averages(events, TRAIN_AVG)
    #    _score_svr(svr, test_events, WINDOW)

    _serialize_svr(svr, WINDOW, sys.argv[-1])


if __name__ == "__main__":
    main()
