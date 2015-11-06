from collections import namedtuple
import scipy.io as sio
import numpy as np
import random
from statsmodels.nonparametric.smoothers_lowess import lowess

Struct = namedtuple("Struct", ["open_pore", "dwell", "pa_blockade", "trace"])

def get_data(filename):
    matrix = sio.loadmat(filename)["Struct"][0][0]
    event_traces = matrix[5]
    num_samples = event_traces.shape[1]

    events = []
    for sample_id in xrange(num_samples):
        dwell = float(matrix[2][0][sample_id])
        pa_blockade = float(matrix[3][0][sample_id])
        open_pore = float(matrix[4][0][sample_id])

        trace = np.array(event_traces[:, sample_id])
        fraction = -pa_blockade / open_pore
        trace = 1 - trace / open_pore
        trace -= min(trace)
        #trace /= fraction
        #print(np.mean(trace), fraction)
        #if dwell > 1:
        #    continue
        #print(pa_blockade, open_pore)

        events.append(Struct(open_pore, dwell, pa_blockade, trace))

    return events


def normalize(signal):
    median = np.median(signal)
    return signal / median


def discretize(signal, num_peaks):
    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * peak_shift
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.mean(signal[left:right]))

    return discrete


def get_consensus(events):
    consensus = None
    for event in events:
        if consensus is None:
            consensus = np.zeros(len(event.trace))
        consensus += event.trace
    return consensus / len(events)


def smooth(signal, frac):
    x = lowess(signal, range(len(signal)), return_sorted=False, frac=frac)
    return x


def get_averages(events, bin_size, flank, reverse=False):
    averages = []
    random.shuffle(events)
    for event_bin in xrange(0, len(events) / bin_size):
        avg_signal = get_consensus(events[event_bin*bin_size:
                                   (event_bin+1)*bin_size])
        if reverse:
            avg_signal = avg_signal[::-1]
        averages.append(avg_signal[flank:-flank])

    return averages
