from collections import namedtuple, defaultdict
import scipy.io as sio
import numpy as np
import random
from copy import deepcopy

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

Struct = namedtuple("Struct", ["fileTag", "StartPoint", "ms_Dwell",
                               "pA_Blockade", "openPore", "eventTrace",
                               "correlation"])
Event = namedtuple("Event", ["trace", "struct"])

def read_mat(filename):
    mat_file = sio.loadmat(filename)
    struct = mat_file["Struct"][0][0]
    peptide = str(mat_file["Peptide"][0])
    event_traces = struct[5]
    num_samples = event_traces.shape[1]

    events = []
    for sample_id in xrange(num_samples):
        file_tag = struct[0][sample_id]
        start_point = float(struct[1][0][sample_id])
        dwell = float(struct[2][0][sample_id])
        pa_blockade = float(struct[3][0][sample_id])
        open_pore = float(struct[4][0][sample_id])
        correlation = float(struct[6][sample_id][0])

        trace = np.array(event_traces[:, sample_id])
        norm_trace = 1 - trace / open_pore
        norm_trace -= min(norm_trace)

        out_struct = Struct(file_tag, start_point, dwell, pa_blockade,
                            open_pore, trace, correlation)
        events.append(Event(norm_trace, out_struct))

    return events, peptide


def write_mat(events, peptide, filename):
    """
    Don't ask why!
    """
    file_tag_arr = np.array(map(lambda e: e.struct.fileTag, events))
    start_arr = np.array(map(lambda e: e.struct.StartPoint, events))
    dwell_arr = np.array(map(lambda e: e.struct.ms_Dwell, events))
    pa_blockade_arr = np.array(map(lambda e: e.struct.pA_Blockade, events))
    open_pore_arr = np.array(map(lambda e: e.struct.openPore, events))
    event_trace_arr = np.array(map(lambda e: e.struct.eventTrace, events))
    corr_arr = np.array(map(lambda e: e.struct.correlation, events))
    matrix = np.array([file_tag_arr, np.array([start_arr]), np.array([dwell_arr]),
                      np.array([pa_blockade_arr]), np.array([open_pore_arr]),
                      np.transpose(event_trace_arr),
                      np.transpose(np.array([corr_arr]))])
    sio.savemat(filename, {"Struct" : np.array([[matrix]]),
                           "Peptide" : peptide})


def normalize(signal):
    median = np.mean(signal - min(signal))
    std = np.std(signal)
    return (signal - min(signal)) / median


"""
def normalize_local(signal, num_aa):
    normalized = []
    WINDOW = len(signal) / num_aa * 10
    window_sum = sum(signal[:WINDOW / 2])
    window_len = WINDOW / 2
    norms = []
    windows = []
    for i in xrange(len(signal)):
        local_mean = float(window_sum) / window_len
        #print(local_mean, window_len)
        normalized.append(signal[i] / local_mean)
        norms.append(local_mean)
        windows.append(window_len)
        #move right end
        if i + WINDOW / 2 + 1 < len(signal):
            window_sum += signal[i + WINDOW / 2 + 1]
            window_len += 1
        #move left end
        if i - WINDOW / 2 > 0:
            window_sum -= signal[i - WINDOW / 2]
            window_len -= 1

    return np.array(normalized)
"""

def cluster_events(events, flank):
    NUM_FEATURES = 1000
    feature_mat = []
    for event in events:
        features = discretize(event.trace[flank:-flank], NUM_FEATURES)
        feature_mat.append(features)

    feature_mat = np.array(feature_mat)
    labels = AffinityPropagation(damping=0.5).fit_predict(feature_mat)
    #labels = KMeans(n_clusters=len(events) / 5).fit_predict(feature_mat)

    by_cluster = defaultdict(list)
    for event, clust_id in enumerate(labels):
        by_cluster[clust_id].append(events[event])
    return map(lambda e: get_consensus(e, flank), by_cluster.values())


def discretize(signal, num_peaks):
    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * peak_shift
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.median(signal[left:right]))

    return discrete


def get_consensus(events, flank):
    matrix = np.array(map(lambda e: e.trace[flank:-flank], events))
    medians = np.mean(matrix, axis=0)
    return medians


def smooth(signal, frac):
    x = lowess(signal, range(len(signal)), return_sorted=False, frac=frac)
    return x


def get_averages(events, bin_size, flank, reverse=False):
    print(len(events))
    averages = []
    events = deepcopy(events)
    random.shuffle(events)
    for event_bin in xrange(0, len(events) / bin_size):
        avg_signal = get_consensus(events[event_bin*bin_size:
                                   (event_bin+1)*bin_size], flank)
        if reverse:
            avg_signal = avg_signal[::-1]
        averages.append(avg_signal)

    return averages
