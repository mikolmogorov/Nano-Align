#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import nanopore.signal_proc as sp


def main():
    events = sp.read_mat(sys.argv[1])
    #sp.normalize(events)

    NUM_FEATURES = 100
    feature_mat = []
    for event in events:
        event.eventTrace = (event.eventTrace - np.mean(event.eventTrace)) / np.std(event.eventTrace)
        features = sp.discretize(event.eventTrace, NUM_FEATURES)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)
    #distances = pdist(feature_mat, metric="euclidean")
    hier = hierarchy.linkage(feature_mat, method="average", metric="correlation")

    def llf(lid):
        return str(len(events[lid].peptide))
    hierarchy.dendrogram(hier, leaf_label_func=llf, leaf_font_size=10)
    plt.show()
    #labels = AffinityPropagation(damping=0.5).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=3).fit_predict(feature_mat)

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(events[event])

    clusters = []
    for cl_events in by_cluster.values():
        print("cluster")
        for event in cl_events:
            print("\t", len(event.peptide))


if __name__ == "__main__":
    main()
