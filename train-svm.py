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
    events, prot = sp.read_mat(sys.argv[1])
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

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(events[event])

    for clust_id, events in by_cluster.items():
        print(len(events))
        consensus = sp.get_consensus(events, 50)
        consensus = sp.normalize(consensus)
        plt.plot(consensus)
        plt.show()


if __name__ == "__main__":
    main()
