#!/usr/bin/env python

"""
De novo nanopore sequencing
"""
from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
from sklearn import svm

from nanopore import get_acids_positions, get_data


def process_event(event_trace, peptide, window_size):
    NUM_FEATURES = 10
    acids_pos = get_acids_positions(peptide, window_size, len(event_trace))
    peak_shift = acids_pos[1] - acids_pos[0]
    features = defaultdict(list)

    for i in xrange(window_size, len(peptide) - window_size):
        acid = peptide[i]
        peak_pos = acids_pos[i]
        window_start = peak_pos - peak_shift * window_size
        window_end = peak_pos + peak_shift * window_size

        feature_indexes = np.linspace(window_start, window_end, NUM_FEATURES)
        acid_features = map(lambda i: event_trace[int(i)], feature_indexes)

        median = np.median(acid_features)
        stddev = np.std(acid_features)
        norm_acid_features = map(lambda i: (i - median) / stddev, acid_features)
        features[acid].append(np.array(norm_acid_features))

    return features


def acid_enum(aa):
    if aa not in acid_enum.ids:
        acid_enum.ids[aa] = acid_enum.next_id
        acid_enum.next_id += 1
    return acid_enum.ids[aa]
acid_enum.next_id = 0
acid_enum.ids = {}


def extract_features(events, peptide, window_size):
    features = defaultdict(list)

    for event in events:
        #normalize event
        #median = np.median(event.trace)
        #std_dev = np.std(event.trace)
        #normalized_event = np.array(map(lambda t: (t - median) / std_dev,
        #                            event.trace))
        event_features = process_event(event.trace, peptide, window_size)
        for acid, acid_features in event_features.items():
            features[acid].extend(acid_features)

    #create feature and label matrix
    features_mat = []
    labels_mat = []
    for aa, aa_features in features.items():
        for feature in aa_features:
            features_mat.append(feature)
            labels_mat.append(acid_enum(aa))
    features_mat = np.array(features_mat)
    labels_mat = np.array(labels_mat)

    return features_mat, labels_mat


def split_datasets(features, labels, precent):
    shuffled_indexes = range(len(labels))
    random.shuffle(shuffled_indexes)
    len_train = int(len(labels) * precent)

    train_features = []
    train_labels = []
    for i in xrange(len_train):
        train_features.append(features[shuffled_indexes[i], :])
        train_labels.append(labels[shuffled_indexes[i]])

    test_features = []
    test_labels = []
    for i in xrange(len_train, len(labels)):
        test_features.append(features[shuffled_indexes[i], :])
        test_labels.append(labels[shuffled_indexes[i]])

    return (np.array(train_features), np.array(train_labels),
            np.array(test_features), np.array(test_labels))


# CCL5
PROT = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
# CXCL1
# PROT = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
# H3N
# PROT = "ARTKQTARKSTGGKAPRKQL"
WINDOW = 3


def main():
    if len(sys.argv) != 2:
        print("Usage: plot.py mat_file")
        return 1

    events = get_data(sys.argv[1])
    features, labels = extract_features(events, PROT, WINDOW)

    train_features, train_labels, test_features, test_labels \
                            = split_datasets(features, labels, 0.7)

    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    print(clf.score(train_features, train_labels))


if __name__ == "__main__":
    main()
