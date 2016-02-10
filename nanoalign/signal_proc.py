#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Some functions for blockade signal processing
"""

from __future__ import print_function
import sys
from collections import namedtuple, defaultdict
import numpy as np
import random
from copy import deepcopy

#from sklearn.cluster import AffinityPropagation
from nanoalign.blockade import BlockadeCluster


def preprocess_blockades(blockades, cluster_size=10,
                         min_dwell=0.5, max_dwell=20):
    """
    The main function for blockade preprocessing.
    Does all preparations and output blockade clusters.
    """
    filtered = _filter_by_duration(blockades, min_dwell, max_dwell)
    frac_current = _fractional_blockades(filtered)
    clusters = _random_cluster(frac_current, cluster_size)
    for cl in clusters:
        cl.consensus = _normalize(_trim_flank_noise(cl.consensus))

    return clusters


def discretize(signal, protein_length):
    """
    Discretizes the signal assuming the given protein length
    """
    WINDOW = 4
    num_peaks = protein_length + WINDOW - 1

    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * (peak_shift - 1)
        #discrete.append(signal[signal_pos])
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.mean(signal[left:right]))

    return discrete


def _filter_by_duration(blockades, min_time, max_time):
    """
    Filters blockades by dwell duration
    """
    new_blockades = list(filter(lambda e: min_time <= e.ms_Dwell <= max_time,
                             blockades))
    filtered_prc = (100 * float(len(blockades) - len(new_blockades)) /
                    len(blockades))
    #print("Filtered by duration: {0:5.2f}%".format(filtered_prc),
    #      file=sys.stderr)
    return new_blockades


def _trim_flank_noise(signal):
    """
    Trims noisy flanking region
    """
    WINDOW = int(0.01 * len(signal))
    def find_local_minima(pos_iter):
        max_good = 0
        max_pos = iter(pos_iter)
        prev_good = False
        for pos in pos_iter:
            left = signal[pos - WINDOW / 2: pos]
            right = signal[pos: pos + WINDOW / 2]
            left_good = len(filter(lambda d: d > signal[pos], left))
            right_good = len(filter(lambda d: d > signal[pos], right))
            score = left_good + right_good

            if max_good < score:
                max_good = score
                max_pos = pos
            good = score > 0.7 * WINDOW
            if good:
                prev_good = True
            else:
                if prev_good:
                    break
        return max_pos

    left = find_local_minima(xrange(WINDOW / 2, int(0.05 * len(signal))))
    right = find_local_minima(xrange(len(signal) - WINDOW / 2,
                                     int(0.95 * len(signal)), -1))

    return signal[left : right]


def _fractional_blockades(blockades):
    """
    Converts blockades curents to fractional values
    """
    blockades = deepcopy(blockades)
    for blockade in blockades:
        if np.median(blockade.eventTrace) < 0:
            blockade.eventTrace = 1 - blockade.eventTrace / blockade.openPore
        else:
            blockade.eventTrace = -blockade.eventTrace / blockade.openPore

    return blockades


def _get_consensus(signals):
    """
    Calculates consensus of multiple signals
    """
    matrix = np.array(map(lambda e: e.eventTrace, signals))
    medians = np.mean(matrix, axis=0)
    return medians


def _normalize(signal):
    """
    Signal normalization
    """
    return (signal - np.mean(signal)) / np.std(signal)


def _random_cluster(blockades, bin_size):
    """
    Randomly splits blockades into clusters and calculates a consensus
    """
    averages = []
    blockades = deepcopy(blockades)
    if bin_size > 1:
        random.shuffle(blockades)
    for event_bin in xrange(0, len(blockades) / bin_size):
        cl_blockades = blockades[event_bin*bin_size : (event_bin+1)*bin_size]
        avg_signal = _get_consensus(cl_blockades)
        averages.append(BlockadeCluster(avg_signal, cl_blockades))

    return averages


"""
def cluster_events(events):
    NUM_FEATURES = 100
    feature_mat = []
    for event in events:
        features = discretize(event.eventTrace, NUM_FEATURES)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)

    labels = AffinityPropagation(damping=0.5).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=3).fit_predict(feature_mat)

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(events[event])

    clusters = []
    for cl_events in by_cluster.values():
        clusters.append(EventCluster(get_consensus(cl_events),
                                     cl_events))
    return clusters
"""
