from __future__ import print_function

from collections import namedtuple, defaultdict
import scipy.io as sio
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import distance

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans


class Struct(object):
    def __init__(self, fileTag, StartPoint, ms_Dwell, pA_Blockade, openPore,
                 eventTrace, correlation, peptide):
        self.fileTag = fileTag
        self.StartPoint = StartPoint
        self.ms_Dwell = ms_Dwell
        self.pA_Blockade = pA_Blockade
        self.openPore = openPore
        self.eventTrace = eventTrace
        self.correlation = correlation
        self.peptide = peptide

        self.norm_trace = None
        self.discr_trace = None


EventCluster = namedtuple("EventCluster", ["consensus", "events"])


def read_mat(filename):
    mat_file = sio.loadmat(filename)
    struct = mat_file["Struct"][0][0]
    event_traces = struct["eventTrace"]
    num_samples = event_traces.shape[1]

    events = []
    for sample_id in xrange(num_samples):
        file_tag = struct["fileTag"][sample_id]
        start_point = float(struct["StartPoint"].squeeze()[sample_id])
        dwell = float(struct["ms_Dwell"].squeeze()[sample_id])
        pa_blockade = float(struct["pA_Blockade"].squeeze()[sample_id])
        open_pore = float(struct["openPore"].squeeze()[sample_id])
        correlation = float(struct["correlation"].squeeze()[sample_id])
        try:
            peptide = str(struct["peptide"][sample_id]).strip()
        except IndexError:
            peptide = None

        trace = np.array(event_traces[:, sample_id])

        out_struct = Struct(file_tag, start_point, dwell, pa_blockade,
                            open_pore, trace, correlation, peptide)
        events.append(out_struct)

    return events


def write_mat(events, filename):
    dtype = [("fileTag", "O"), ("StartPoint", "O"), ("ms_Dwell", "O"),
             ("pA_Blockade", "O"), ("eventTrace", "O"), ("openPore", "O"),
             ("correlation", "O"), ("peptide", "O")]
    file_tag_arr = np.array(map(lambda e: e.fileTag, events))
    peptide_arr = np.array(map(lambda e: e.peptide, events))
    start_arr = np.array(map(lambda e: e.StartPoint, events))
    dwell_arr = np.array(map(lambda e: e.ms_Dwell, events))
    pa_blockade_arr = np.array(map(lambda e: e.pA_Blockade, events))
    open_pore_arr = np.array(map(lambda e: e.openPore, events))
    event_trace_arr = np.array(map(lambda e: e.eventTrace, events))
    corr_arr = np.array(map(lambda e: e.correlation, events))

    struct = (file_tag_arr, [start_arr], [dwell_arr], [pa_blockade_arr],
              np.transpose(event_trace_arr), [open_pore_arr],
              [corr_arr], peptide_arr)
    sio.savemat(filename, {"Struct" : np.array([[struct]], dtype=dtype)})


def trim_flank_noise(signal):
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

    #plt.plot(signal)
    #plt.plot([left, left], [min(signal), max(signal)])
    #plt.plot([right, right], [min(signal), max(signal)])
    #plt.show()
    return signal[left : right]


def normalize(events):
    pas = []
    ops = []
    ress = []
    for event in events:
        #norm_trace = event.eventTrace - min(event.eventTrace)
        #scale = np.percentile(norm_trace, 75) - np.percentile(norm_trace, 25)
        #event.eventTrace = (norm_trace - np.mean(norm_trace)) / np.std(norm_trace)
        #event.eventTrace = (event.eventTrace - np.mean(event.eventTrace))
        event.eventTrace = 1 - event.eventTrace / event.openPore
        #event.eventTrace -= np.mean(event.eventTrace)
        #event.eventTrace = (event.eventTrace - np.mean(event.eventTrace)) / np.std(event.eventTrace)


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
        #print("cluster")
        #for event in cl_events:
        #    print("\t", len(event.peptide))
    return clusters


def discretize(signal, num_peaks):
    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * (peak_shift - 1)
        #discrete.append(signal[signal_pos])
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.mean(signal[left:right]))

    return discrete


def get_consensus(events):
    matrix = np.array(map(lambda e: e.eventTrace, events))
    medians = np.mean(matrix, axis=0)

    medians = (medians - np.mean(medians)) / np.std(medians)
    return medians


def smooth(signal, frac):
    x = lowess(signal, range(len(signal)), return_sorted=False, frac=frac)
    return x


def get_averages(events, bin_size):
    averages = []
    events = deepcopy(events)
    random.shuffle(events)
    for event_bin in xrange(0, len(events) / bin_size):
        cl_events = events[event_bin*bin_size : (event_bin+1)*bin_size]
        avg_signal = get_consensus(cl_events)
        averages.append(EventCluster(avg_signal, cl_events))

    return averages
