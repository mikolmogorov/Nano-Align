#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import nanopore.signal_proc as sp

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def main():
    events = sp.read_mat(sys.argv[1])
    sp.normalize(events)

    #NUM_FEATURES = 100
    feature_mat = []
    for event in events:
        #event.eventTrace = (event.eventTrace - np.mean(event.eventTrace)) / np.std(event.eventTrace)
        #event.eventTrace += event.openPore
        features = sp.discretize(event.eventTrace, len(event.peptide) + 4)[:24]
        #features = (features - np.mean(features)) / np.std(features)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)
    #distances = pdist(feature_mat, metric="euclidean")
    hier = hierarchy.linkage(feature_mat, method="average", metric="correlation")

    def llf(lid):
        return str(len(events[lid].peptide))
    hierarchy.dendrogram(hier, leaf_label_func=llf, leaf_font_size=10)
    plt.show()

    pca = PCA(2)
    pca.fit(feature_mat)
    new_x = pca.transform(feature_mat)
    colors = map(lambda e: len(e.peptide), events)
    plt.scatter(rand_jitter(new_x[:, 0]), rand_jitter(new_x[:, 1]), s=50,
                alpha=0.5, c=colors)
    plt.show()

    labels = hierarchy.fcluster(hier, 5, criterion="distance")
    #labels = hierarchy.fcluster(hier, 0.9)
    #labels = AffinityPropagation(damping=0.9).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=3).fit_predict(feature_mat)

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(events[event])

    clusters = []
    errors = []
    sizes = []
    for cl_events in by_cluster.values():
        print("cluster")
        hist = defaultdict(int)
        sizes.append(len(cl_events))
        for event in cl_events:
            print("\t", len(event.peptide))
            hist[len(event.peptide)] += 1

        if len(cl_events) > 5:
            error = 0
            major = max(hist, key=hist.get)
            for pep in hist:
                if pep != major:
                    error += hist[pep]
            errors.append(float(error) / len(cl_events))

    print("Clusters: {0}".format(len(by_cluster)))
    print("Median size: {0}".format(np.median(sizes)))
    print("Average error: {0}".format(np.mean(errors)))
    print("Median error: {0}".format(np.median(errors)))


if __name__ == "__main__":
    main()
