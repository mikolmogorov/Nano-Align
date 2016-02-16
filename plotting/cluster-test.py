#!/usr/bin/env python

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Clustering testing
"""

from __future__ import print_function
import sys
from collections import defaultdict
from itertools import product

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.mixture import GMM
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import distance
from scipy import signal
import math

import nanopore.signal_proc as sp


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

def peptide_color_label(peptide):
    if peptide.startswith("MARTKQ"):
        return "red", "H32"
    elif peptide.startswith("MSGRGK"):
        return "green", "H4"
    elif peptide.startswith("LQKRP"):
        return "yellow", "H3"
    elif peptide.startswith("SPYSSD"):
        return "blue", "CCL5"
    else:
        return None


def draw_pca(feature_mat, clusters):
    pca = PCA(2)
    pca.fit(feature_mat)
    new_x = pca.transform(feature_mat)

    by_peptide = defaultdict(list)
    for i in xrange(len(clusters)):
        by_peptide[clusters[i].events[0].peptide].append(new_x[i])

    fig = plt.subplot()
    for peptide, proj in by_peptide.items():
        proj = np.array(proj)
        color, label = peptide_color_label(peptide)
        fig.scatter(rand_jitter(proj[:, 0]), rand_jitter(proj[:, 1]), s=50,
                    alpha=0.5, c=color, label=label)

    #red_patch = mpatches.Patch(color="red", label="H32")
    #green_patch = mpatches.Patch(color="green", label="H4")
    #yellow_patch = mpatches.Patch(color="yellow", label="H3")
    #blue_patch = mpatches.Patch(color="blue", label="CCL5")

    fig.grid(True)
    fig.legend()
    plt.show()


def gmm_cluster(feature_mat, clusters):
    gmm = GMM(n_components=3, covariance_type="tied")
    labels = gmm.fit_predict(feature_mat)

    pca = PCA(2)
    pca.fit(feature_mat)
    new_x = pca.transform(feature_mat)

    proj_means = pca.transform(gmm.means_)

    by_peptide = defaultdict(list)
    for i in xrange(len(clusters)):
        by_peptide[clusters[i].events[0].peptide].append(new_x[i])

    fig = plt.subplot()
    for peptide, proj in by_peptide.items():
        proj = np.array(proj)
        color, label = peptide_color_label(peptide)
        fig.scatter(rand_jitter(proj[:, 0]), rand_jitter(proj[:, 1]), s=50,
                    alpha=0.5, c=color, label=label)

    fig.scatter(proj_means[:, 0], proj_means[:, 1], marker="x", c="black")

    fig.legend()
    plt.show()
    return labels


def cluster_test(events):
    sp.normalize(events)
    events = sp.filter_by_time(events, 1.0, 20.0)

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
                                 len(cluster.events[0].peptide) + 3)[:23]
        #features = sp.discretize(sp.trim_flank_noise(cluster.consensus), NUM_FEATURES)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)

    hier = hierarchy.linkage(feature_mat, method="average", metric="euclidean")

    def llf(lid):
        return peptide_color_label((clusters[lid].events[0].peptide))[1]
    hierarchy.dendrogram(hier, leaf_label_func=llf, leaf_font_size=10)
    plt.show()

    #draw_pca(feature_mat, clusters)

    #labels = hierarchy.fcluster(hier, 0.4, criterion="distance")
    #labels = hierarchy.fcluster(hier, 0.95)
    #labels = AffinityPropagation(damping=0.9).fit_predict(feature_mat)
    #labels = DBSCAN(eps=0.001, min_samples=20).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=3).fit_predict(feature_mat)
    labels = gmm_cluster(feature_mat, clusters)

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
            print("\t", peptide_color_label(cluster.events[0].peptide)[1])
            hist[peptide_color_label(cluster.events[0].peptide)[1]] += 1

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
