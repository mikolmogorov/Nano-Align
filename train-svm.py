#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN

import nanopore.signal_proc as sp


def main():
    events = sp.read_mat(sys.argv[1])
    feature_mat = []
    for event in events:
        features = sp.discretize(event.trace, 1000)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)
    num_clusters = len(events) / 5

    kmeans = AffinityPropagation(damping=0.5)
    #kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(feature_mat)
    #random.shuffle(labels)

    hist = defaultdict(int)
    for l in labels:
        hist[l] += 1
    for l in sorted(hist):
        print(l, hist[l])

    consensuses = defaultdict(lambda: np.zeros(len(events[0].trace)))
    for event, clust_id in enumerate(labels):
        consensuses[clust_id] += events[event].trace
    for cust_id, cons in consensuses.items():
        cons /= hist[clust_id]

    for clust_id, cons in consensuses.items():
        print(hist[clust_id])
        plt.plot(cons[20:-20])
        plt.show()


if __name__ == "__main__":
    main()
