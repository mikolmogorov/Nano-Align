#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import defaultdict
from itertools import product

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
import math

import nanopore.signal_proc as sp

from estimate_length import gcd_features

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def distance_test(events):
    #sp.normalize(events)

    #NUM_PTS = 50
    for pts in xrange(10, 300):
        feature_mat = []
        for event in events:
            features = sp.discretize(event.eventTrace, pts)
            features = (features - np.mean(features)) / np.std(features)
            feature_mat.append(features)

        distances = []
        for event_1, event_2 in product(feature_mat, repeat=2):
            #d = float(distance.cityblock(event_1, event_2)) / pts
            d = 1 - distance.correlation(event_1, event_2)
            distances.append(d)

        print(pts, np.median(distances))
        #plt.hist(distances, bins=100)
        #plt.show()


def cluster_test(events):
    sp.normalize(events)
    #events = sp.filter_by_time(events, 1.0, 5.0)

    NUM_FEATURES = 50
    feature_mat = []

    by_peptide = defaultdict(list)
    for event in events:
        by_peptide[event.peptide].append(event)
    clusters = []
    for peptide in by_peptide:
        clusters.extend(sp.get_averages(by_peptide[peptide], 1))

    for cluster in clusters:
        features = sp.discretize(cluster.consensus,
                                 len(cluster.events[0].peptide) + 4)[:24]
        #xx, features = peak_features(cluster.consensus, 20)
        features = gcd_features(cluster.consensus)
        #features = get_spectra(cluster.consensus)
        #features = characteristic_peaks(cluster.consensus)
        #print(cluster.events[0].peptide)
        #f, den = signal.periodogram(cluster.consensus)
        #print(len(f))
        #plt.semilogy(den[:1000])
        #plt.show()
        #features = sp.discretize(sp.trim_flank_noise(cluster.consensus), NUM_FEATURES)
        #features = np.fft.fft(cluster.consensus, 100)
        #features = den[:500]
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)
    #distances = np.zeros((len(clusters), len(clusters)))
    #for i in xrange(len(clusters)):
    #    for j in xrange(len(clusters)):
    #        if i != j:
    #            distances[i][j] = score(feature_mat[i], feature_mat[j])

    hier = hierarchy.linkage(feature_mat, method="average", metric="euclidean")

    def llf(lid):
        return str(len(clusters[lid].events[0].peptide))
    hierarchy.dendrogram(hier, leaf_label_func=llf, leaf_font_size=10)
    plt.show()

    pca = PCA(2)
    pca.fit(feature_mat)
    new_x = pca.transform(feature_mat)
    colors = map(lambda c: len(c.events[0].peptide), clusters)
    fig = plt.subplot()
    #fig.set_xscale("log")
    #fig.set_yscale("log")
    fig.scatter(rand_jitter(new_x[:, 0]), rand_jitter(new_x[:, 1]), s=50,
                alpha=0.5, c=colors)
    fig.grid(True)
    plt.show()

    #labels = hierarchy.fcluster(hier, 0.4, criterion="distance")
    #labels = hierarchy.fcluster(hier, 0.95)
    labels = AffinityPropagation(damping=0.9).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=3).fit_predict(feature_mat)

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(clusters[event])

    clusters = []
    errors = []
    sizes = []
    for cl_events in by_cluster.values():
        print("cluster")
        hist = defaultdict(int)
        sizes.append(len(cl_events))
        for cluster in cl_events:
            print("\t", len(cluster.events[0].peptide))
            hist[len(cluster.events[0].peptide)] += 1

        if len(cl_events) > 1:
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


def main():
    events = sp.read_mat(sys.argv[1])
    #distance_test(events)
    cluster_test(events)


if __name__ == "__main__":
    main()
