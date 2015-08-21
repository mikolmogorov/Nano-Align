#!/usr/bin/env python

from __future__ import print_function
import sys
from collections import namedtuple

import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

Struct = namedtuple("Struct", ["open_current", "dwell", "pa_blockade", "trace"])


def get_data(filename):
    matrix = sio.loadmat(filename)["Struct"][0][0]
    event_traces = matrix[5]
    num_samples = event_traces.shape[1]

    events = []
    for sample_id in xrange(num_samples):
        dwell = float(matrix[2][0][sample_id])
        pa_blockade = float(matrix[3][0][sample_id])
        open_current = float(matrix[4][0][sample_id])

        trace = np.array(event_traces[:, sample_id])
        trace = -(trace - open_current) / open_current

        events.append(Struct(open_current, dwell, pa_blockade, trace))

    return events


def find_peaks(signal):
    WINDOW = 6
    deriv = np.zeros(len(signal) - 2)
    for i in xrange(len(deriv)):
        deriv[i] = (signal[i + 2] - signal[i]) / 2

    peaks = []
    for pos in xrange(WINDOW / 2, len(deriv) - WINDOW / 2):
        left = deriv[pos - WINDOW / 2: pos]
        right = deriv[pos: pos + WINDOW / 2]

        if all(x > 0 for x in left) and all(y < 0 for y in right):
            peaks.append(pos)

    return peaks


def theoretical_signal(peptide, window_size):
    VOLUMES = {"I": 0.1688, "F": 0.2034, "V": 0.1417, "L": 0.1679,
               "W": 0.2376, "M": 0.1708, "A": 0.0915, "G": 0.0664,
               "C": 0.1056, "Y": 0.2036, "P": 0.1293, "T": 0.1221,
               "S": 0.0991, "H": 0.1673, "E": 0.1551, "N": 0.1359,
               "Q": 0.1611, "D": 0.1245, "K": 0.1713, "R": 0.2021}

    signal = []
    for i in xrange(-window_size + 1, len(peptide) - 1):
        start, end = max(i, 0), min(i + window_size, len(peptide))
        volume = sum(map(VOLUMES.get, peptide[start:end]))
        signal.append(volume / window_size)
    return signal


def get_acids_positions(peptide, window_size, plot_len):
    num_peaks = len(peptide) + window_size - 1
    peak_shift = float(plot_len) / (num_peaks - 1)
    initial_shift = (window_size - 1) * peak_shift / 2
    positions = []
    for aa in xrange(len(peptide)):
        positions.append(initial_shift + aa * peak_shift)
    return positions


def fill_gaps(alignment):
    res = np.zeros(len(alignment))
    open_gap = None if alignment[0] is not None else 0

    for i in xrange(len(alignment)):
        if open_gap is not None:
            if alignment[i] is not None:
                left, right = alignment[open_gap], alignment[i]
                if left is None:
                    left = right

                for j in xrange(open_gap, i + 1):
                    rate = (j - open_gap) / (i - open_gap)
                    res[j] = left + (right - left) * rate
                open_gap = None
        else:
            if alignment[i] is not None:
                res[i] = alignment[i]
            else:
                open_gap = i - 1
    return res


def fit_to_model(model_trace, event_trace):
    match = lambda p1, p2: 100 * (0.1 - abs(p1 - p2))
    score, aln_model, aln_event = glob_affine_gap(model_trace, event_trace,
                                                  -4, -3, match)
    filled_event = fill_gaps(aln_event)
    #print(len(aln_model), len(aln_event))
    #print(len([x for x in aln_model if x is None]))
    #print(len([x for x in aln_event if x is None]))
    trimmed_event = []
    for i in xrange(len(filled_event)):
        if aln_model[i] is not None:
            trimmed_event.append(filled_event[i])
    return trimmed_event


def alignment(signal_1, signal_2):
    match = lambda p1, p2: 100 * (0.1 - abs(p1 - p2))
    score, aln_1, aln_2 = glob_affine_gap(signal_1, signal_2, -4, -3, match)
    return score, fill_gaps(aln_1), fill_gaps(aln_2)


def glob_affine_gap(seq1, seq2, gap_open, gap_ext, match_fun):
    len1 = len(seq1)
    len2 = len(seq2)

    s_m = np.ones((len1 + 1, len2 + 1)) * float("-inf")
    s_x = np.ones((len1 + 1, len2 + 1)) * float("-inf")
    s_y = np.ones((len1 + 1, len2 + 1)) * float("-inf")
    b_m = np.zeros((len1 + 1, len2 + 1))
    b_x = np.zeros((len1 + 1, len2 + 1))
    b_y = np.zeros((len1 + 1, len2 + 1))
    s_m[0][0] = 0

    for i in xrange(len1 + 1):
        s_x[i][0] = gap_open + (i - 1) * gap_ext
        b_x[i][0] = 1
    for i in xrange(len2 + 1):
        s_y[0][i] = gap_open + (i - 1) * gap_ext
        b_y[0][i] = 2

    for i in xrange(1, len1 + 1):
        for j in xrange(1, len2 + 1):
            delta = match_fun(seq1[i - 1], seq2[j - 1])

            lst_m = [s_m[i - 1][j - 1] + delta,
                     s_x[i - 1][j - 1] + delta,
                     s_y[i - 1][j - 1] + delta]
            lst_x = [s_m[i - 1][j] + gap_open,
                     s_x[i - 1][j] + gap_ext,
                     s_y[i - 1][j] + gap_open]
            lst_y = [s_m[i][j - 1] + gap_open,
                     s_x[i][j - 1] + gap_open,
                     s_y[i][j - 1] + gap_ext]

            s_m[i][j] = max(lst_m)
            s_x[i][j] = max(lst_x)
            s_y[i][j] = max(lst_y)

            b_m[i][j] = lst_m.index(s_m[i][j])
            b_x[i][j] = lst_x.index(s_x[i][j])
            b_y[i][j] = lst_y.index(s_y[i][j])

    # backtracking
    all_mat = [s_m, s_x, s_y]
    i, j = len1, len2
    cur_mat = max(s_m, s_x, s_y, key=lambda x: x[len1][len2])
    score = cur_mat[len1][len2]
    res1, res2 = [], []
    while i > 0 or j > 0:
        if id(cur_mat) == id(s_m):
            res1.append(seq1[i - 1])
            res2.append(seq2[j - 1])
            cur_mat = all_mat[int(b_m[i][j])]
            i -= 1
            j -= 1
        elif id(cur_mat) == id(s_x):
            res1.append(seq1[i - 1])
            res2.append(None)
            cur_mat = all_mat[int(b_x[i][j])]
            i -= 1
        elif id(cur_mat) == id(s_y):
            res1.append(None)
            res2.append(seq2[j - 1])
            cur_mat = all_mat[int(b_y[i][j])]
            j -= 1
    return score, res1[::-1], res2[::-1]


def compare_events(events):
    event_len = len(events[0].trace)
    for event_1, event_2 in zip(events[:-1], events[1:]):
        median_1 = np.median(event_1.trace)
        median_2 = np.median(event_2.trace)
        std_1 = np.std(event_1.trace)
        std_2 = np.std(event_2.trace)

        scaled_2 = map(lambda t: (t - median_2) * (median_1 / median_2) + median_1,
                       event_2.trace)

        reduced_1 = map(lambda i: event_1.trace[i], xrange(0, event_len, 10))
        reduced_2 = map(lambda i: scaled_2[i], xrange(0, event_len, 10))
        score, aligned_1, aligned_2 = alignment(reduced_1, reduced_2)

        plt.figure(dpi=160)
        plt.plot(aligned_1)
        plt.plot(aligned_2)
        plt.show()


def plot_blockades(events, prot, window):
    event_len = len(events[0].trace)
    num_samples = len(events)

    model_volume = theoretical_signal(prot, window)
    model_volume_mean = np.mean(model_volume)
    model_volume_std = np.std(model_volume)
    model_grid = [i * event_len / (len(model_volume) - 1)
                  for i in xrange(len(model_volume))]
    # consensus = get_consensus(events)

    print("Number of samles: {0}".format(num_samples))
    for event in events:
        # peaks = find_peaks(event.trace)
        # print("Peaks detected: {0}".format(len(peaks)))

        sample_mean = np.mean(event.trace)
        sample_std = np.std(event.trace)
        scale_factor = sample_std / model_volume_std

        # fitting
        model_scaled = map(lambda t: (t - model_volume_mean) * scale_factor + sample_mean,
                           model_volume)
        interp_fun = interp1d(model_grid, model_scaled, kind="cubic")
        model_interp = interp_fun(xrange(event_len))
        ###

        reduced_trace = map(lambda i: event.trace[i], xrange(0, event_len, 10))
        reduced_model = map(lambda i: model_interp[i], xrange(0, event_len, 10))
        fitted_event = fit_to_model(reduced_model, reduced_trace)
        #score, aligned_trace, aligned_model = alignment(reduced_trace,
        #                                                reduced_model)
        #inverted_score, _a_t, _a_m = alignment(reduced_trace, reduced_model[::-1])

        print("Dwell: {0}".format(event.dwell))
        print("Open current: {0}".format(event.open_current))
        print("Blockade shift: {0}".format(event.pa_blockade))
        # print("Match score: {0}".format(score))
        # print("Inverted model score: {0}".format(inverted_score))

        plt.figure(dpi=160)
        #plt.plot(aligned_trace, label="blockade")
        #plt.plot(aligned_model, label="model")
        plt.plot(fitted_event, label="blockade")
        plt.plot(reduced_model, label="model")

        # adding AAs text:
        acids_pos = get_acids_positions(prot, window, len(fitted_event))
        #acids_pos = get_acids_positions(prot, window, len(event.trace))
        for i, aa in enumerate(prot):
            plt.text(acids_pos[i], 0.1, aa, fontsize=10)

        # plt.plot(consensus, label="consenus")
        # yy = map(lambda p: event.trace[p], peaks)
        # plt.plot(peaks, yy, "ro")

        plt.legend()
        plt.show()


def get_consensus(events):
    consensus = None
    for event in events:
        if consensus is None:
            consensus = np.zeros(len(event.trace))
        consensus += event.trace
    return consensus / len(events)


# CCL5
PROT = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
# CXCL1
# PROT = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
# H3N
# PROT = "ARTKQTARKSTGGKAPRKQL"
WINDOW = 3


def main():
    if len(sys.argv) != 2:
        print("Usage: plot.py mat_file")
        return 1

    events = get_data(sys.argv[1])
    #plot_blockades(events, PROT, WINDOW)
    compare_events(events)


if __name__ == "__main__":
    main()
