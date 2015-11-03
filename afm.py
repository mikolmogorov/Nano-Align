#!/usr/bin/env python


import sys
import os
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from statsmodels.nonparametric.smoothers_lowess import lowess

from nanopore import theoretical_signal

Struct = namedtuple("Struct", ["open_pore", "final_current", "theory", "trace"])

def _discretize(signal, num_peaks):
    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * peak_shift
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.mean(signal[left:right]))

    return discrete


def get_data(filenames):
    events = []
    for filename in filenames:
        struct_name = os.path.basename(filename[:filename.index(".")])
        matrix = sio.loadmat(filename)[struct_name][0][0]
        #event_traces = matrix[5]
        #num_samples = event_traces.shape[1]

        #dwell = float(matrix[2][0][sample_id])
        #pa_blockade = float(matrix[3][0][sample_id])
        open_pore = float(matrix[0][0])
        trace = np.array(matrix[1][:, 0])
        final_current = np.array(matrix[2][:, 0])
        theory = np.array(matrix[3][:, 0])
        #fraction = -pa_blockade / open_pore
        #    trace = 1 - trace / open_pore
        #    trace -= min(trace)
            #trace /= fraction
            #print(np.mean(trace), fraction)

        events.append(Struct(open_pore, final_current, theory, trace))
        #x = lowess(trace, range(len(trace)), return_sorted=False, frac=(float(1)/136))
        plt.plot(final_current)
        plt.plot(theory)
        plt.show()

    return events


def main():
    events = get_data(sys.argv[1:])


if __name__ == "__main__":
    main()
