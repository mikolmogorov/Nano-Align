#!/usr/bin/env python


from __future__ import print_function

import sys
import math
from collections import defaultdict

import numpy as np
import nanopore.signal_proc as sp
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from scipy.signal import find_peaks_cwt, periodogram
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn import mixture
from scipy import signal


def find_peaks(signal):
    signal = np.array(signal)
    WINDOW = 8

    peaks = []
    for pos in xrange(WINDOW / 2, len(signal) - WINDOW / 2):
        left = signal[pos - WINDOW / 2: pos] - signal[pos]
        right = signal[pos + 1: pos + WINDOW / 2 + 1] - signal[pos]

        if np.percentile(left, 95) < 0 and np.percentile(right, 95) < 0:
            peaks.append(pos)

    return peaks


def peak_features(signal, n_peaks, minimum=False, ranged=False):
    signal = np.array(signal)
    WINDOW = 8

    peaks = []
    for pos in xrange(WINDOW / 2, len(signal) - WINDOW / 2):
        left = signal[pos - WINDOW / 2: pos] - signal[pos]
        right = signal[pos + 1: pos + WINDOW / 2 + 1] - signal[pos]

        if not minimum:
            if (left < 0).all() and (right < 0).all():
                peaks.append((pos, abs(np.mean(left) + np.mean(right))))
        else:
            if (left > 0).all() and (right > 0).all():
                #peaks.append((pos, abs(np.mean(left) + np.mean(right))))
                peaks.append((pos, signal[pos]))

    selected = sorted(peaks, key=lambda p: p[1], reverse=not minimum)[:n_peaks]
    xx = map(lambda p: p[0], selected)
    if not ranged:
        xx.sort()
    yy = map(lambda p: signal[p], xx)
    #if len(xx) < n_peaks:
    #    xx.extend([9999] * (n_peaks - len(xx)))
    #    yy.extend([0] * (n_peaks - len(yy)))
    return xx, yy


def plot_dots(signal, n_dots):
    xx = np.linspace(0, 9999, n_dots)
    yy = map(lambda x: signal[x], xx)
    plt.scatter(xx, yy)


def deriv(signal):
    return np.gradient(signal, 2)


def peaks_score(deriv, n_dots, start, end):
    peaks = np.linspace(start, end, n_dots)
    deriv_vals = map(lambda x: deriv[x], peaks)
    return np.std(np.array(deriv_vals) ** 2)


def score(peaks_1, peaks_2):
    score = 0
    for p_1 in peaks_1:
        for p_2 in peaks_2:
            if abs(p_1 - p_2) <= 5:
                score += 1
    return 1 - float(score) / (len(peaks_1) + len(peaks_2) - score)


def get_peak_spectra(signal):
    xx, yy = peak_features(signal, 2000)
    der = deriv(signal)

    vals = []
    r = range(20, 200)
    for i in r:
        vals.append(peaks_score(der, i, xx[0], xx[-1]))
    vals = (vals - min(vals)) / (max(vals) - min(vals))
    #return vals
    return savitsky_golay(vals, 51, 3)


def characteristic_peaks(signal):
    peaks = find_peaks(get_spectra(signal))
    filtered_peaks = []
    for p1 in peaks:
        if all(map(lambda p2: abs(p1 - p2) > 5, filtered_peaks)):
            filtered_peaks.append(p1)
    return filtered_peaks


def savitsky_golay(y, window_size, order, deriv=0, rate=1):
    try:
       window_size = np.abs(np.int(window_size))
       order = np.abs(np.int(order))
    except ValueError, msg:
       raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
       raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
       raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    #precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    #pad the signal at the extremes with
    #values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def smooth(signal):
    return savitsky_golay(signal, 11, 3)


def get_spectra(signal):
    w = fftpack.rfft(sp.trim_flank_noise(signal))
    return savitsky_golay((w ** 2)[:1000], 21, 4)


"""
def get_peaks(signal):
    w = fftpack.rfft(sp.trim_flank_noise(signal))
    #f = fftpack.rfftfreq(len(event))
    #spectrum = w ** 2
    #cutoff_idx = spectrum < (spectrum.max() / 5)
    #print(cutoff_idx)
    #print(spectrum[:20])
    w2 = w.copy()
    w2[:40] = 0
    #w2[1000:] = 0
    smoothed = fftpack.irfft(w2)
    f, (s1, s2) = plt.subplots(2)

    #print(len(find_peaks(signal)), len(find_peaks(smoothed)))

    #spectra = savitsky_golay((w ** 2)[:1000], 21, 4)
    spectra = get_spectra(signal)
    #peaks = find_peaks_cwt(spectra, np.arange(1, len(spectra)))
    peaks = find_peaks(spectra)
    print(peaks)

    autocor = np.correlate(signal, signal, mode="full")
    autocor /= autocor[autocor.argmax()]
    #autocor = estimated_autocorrelation(signal)

    s1.plot(signal)
    s2.plot(spectra)
    s2.set_yscale("log")
    plt.show()
"""


"""
def score_distr(events):
    scores = []

    peaks = map(lambda e: characteristic_peaks(e.eventTrace), events)
    for e_1 in peaks:
        for e_2 in peaks:
            scores.append(score(e_1, e_2))

    plt.hist(scores, bins=100)
    plt.show()
"""


"""
def cross_corr(events):
    by_peptide = defaultdict(list)
    for event in events:
        by_peptide[event.peptide].append(event)
    pps = by_peptide.keys()

    for e_1, e_2 in zip(by_peptide[pps[0]], by_peptide[pps[1]]):
        trace_1 = sp.trim_flank_noise(e_1.eventTrace)
        trace_2 = sp.trim_flank_noise(e_2.eventTrace)

        f, (s1, s2, s3, s4) = plt.subplots(4)
        s1.plot(trace_1)
        s2.plot(get_peak_spectra(trace_1))
        s3.plot(trace_2)
        s4.plot(get_peak_spectra(trace_2))
        #s3.set_yscale("log")
        plt.show()
"""


def gcd_fuzz(numbers):
    vals = []
    for div in xrange(1, 300):
        rems = []
        for num in numbers:
            rem = min(num % div, (num // div + 1) * div - num)
            rems.append((float(rem)))

        vals.append(np.mean(rems) / div)

    #vals = [0] * 50 + vals
    gcd_x, gcd_y = peak_features(vals, 100, minimum=True, ranged=True)
    #gcds = map(lambda x: x + 50)
    gcds = filter(lambda x: 40 < x < 500, gcd_x)
    #if not gcds:
    #    gcds = [-1]

    return vals, gcds


def gcd_features(s):
    xx, yy = peak_features(s, 200)
    diff_1 = sorted(np.array(xx)[1:] - np.array(xx)[:-1])
    diff_2 = sorted(np.array(xx)[2:] - np.array(xx)[:-2])
    diff_3 = sorted(np.array(xx)[3:] - np.array(xx)[:-3])
    diff = diff_1 + diff_2 + diff_3

    #smoothing with Gaussian window
    d_hist, bin_edges = np.histogram(diff, bins=100, range=(50, 500))
    window = signal.gaussian(7, std=3)
    smooth = np.convolve(d_hist, window, mode="same")
    smooth_smooth = savitsky_golay(smooth, 11, 2)

    return np.array(smooth_smooth) / max(smooth_smooth)

    #hist_x, hist_y = peak_features(smooth_smooth, 10)
    #char_peaks = map(lambda x: int(bin_edges[x]), hist_x)

    #return gcd_fuzz(char_peaks)


def draw_plot():
    h32 = "../H32_rev.mat"
    h4 = "../H4_all_rev.mat"
    ccl5 = "../CCL5_rev.mat"
    h3 = "../H3N_rev.mat"

    peaks = []
    for sample in [h32, h4, ccl5, h3]:
        print(sample)
        events = sp.read_mat(sample)
        sample_peaks = []
        for event in events:
            xx, yy = peak_features(event.eventTrace, 2000)
            sample_peaks.append(len(xx) / event.ms_Dwell)

        peaks.append(sample_peaks)

    plt.hist(peaks, bins=20, histtype="bar", normed=1,
             label=["H32", "H4", "CCL5", "H3"])
    plt.legend()
    plt.show()


def show_peaks(events):
    all_peaks = []

    #dwells = []
    #lengths = []
    for event in events:
        xx, yy = peak_features(event.eventTrace, 2000)
        diff_1 = sorted(np.array(xx)[1:] - np.array(xx)[:-1])
        diff_2 = sorted(np.array(xx)[2:] - np.array(xx)[:-2])
        diff_3 = sorted(np.array(xx)[3:] - np.array(xx)[:-3])
        diff = diff_1 + diff_2 + diff_3

        all_peaks.append(len(xx) / event.ms_Dwell)
        continue

        true_peaks = np.linspace(xx[0], xx[-1], len(event.peptide))
        print(len(event.peptide))

        peak_spectra = get_peak_spectra(event.eventTrace)

        #dwells.append(event.ms_Dwell)
        #lengths.append(len(xx))

        #smoothing with Gaussian window
        d_hist, bin_edges = np.histogram(diff, bins=100, range=(50, 500))
        window = signal.gaussian(21, std=1)
        smooth = np.convolve(d_hist, window, mode="same")
        smooth_smooth = savitsky_golay(smooth, 5, 3)

        #autocor = np.correlate(smooth_smooth, smooth_smooth, mode="full")
        #autocor /= autocor[autocor.argmax()]

        hist_x, hist_y = peak_features(smooth_smooth, 10)
        char_peaks = map(lambda x: int(bin_edges[x + 1]), hist_x)

        cd = np.median(np.array(char_peaks[1:]) - np.array(char_peaks[:-1]))
        print(event.ms_Dwell, char_peaks, cd)
        gcd_plot, gcds = gcd_fuzz(char_peaks)
        print("gcd", gcds)

        fft = fftpack.rfft(smooth_smooth)
        #length = gcds[0]

        f, (s1, s2, s3, s4, s5) = plt.subplots(5)
        s1.plot(event.eventTrace)
        #s1.scatter(xx, yy)
        s1.scatter(true_peaks, map(lambda x: event.eventTrace[x], true_peaks))

        s2.plot(bin_edges[1:], smooth_smooth)
        s3.plot(range(20, len(peak_spectra) + 20), peak_spectra)
        #s5.plot(range(50, 300), gcds)
        #s4.plot(bin_edges[1:], smooth_smooth)
        s4.plot(gcd_plot)
        s5.plot(bin_edges[1:], fft ** 2)
        s5.set_yscale("log")
        plt.show()

        if gcds:
            all_peaks.append(gcds[0] / ms_Dwell)
        #if not math.isnan(cd):
        #    all_peaks.append(cd)

    print(np.mean(all_peaks), np.std(all_peaks))
    plt.hist(all_peaks, bins=100)
    #plt.scatter(dwells, lengths)
    plt.show()


def main():
    #draw_plot()
    events = sp.read_mat(sys.argv[1])
    events = sp.filter_by_time(events, 1.0, 30.0)
    sp.normalize(events)

    #cross_corr(events)
    #score_distr(events)
    show_peaks(events)

    for event in events:
        print(event.ms_Dwell)

        #get_peaks(event.eventTrace)
        #continue

        trace = sp.trim_flank_noise(event.eventTrace)
        smooth_signal = smooth(trace)

        vals = get_spectra(trace)
        #print(characteristic_peaks(trace))
        #print event.ms_Dwell, map(lambda p: p + 20, find_peaks(vals))
        #smooth_vals = get_peak_spectra(smooth_signal)
        frqs, dens = periodogram(smooth_signal)
        print(characteristic_peaks(smooth_signal))

        f, (s1, s2, s3, s4) = plt.subplots(4)
        s1.plot(trace)
        #s2.plot(range(20, 200), vals)
        s2.plot(vals)
        s2.set_yscale("log")
        s3.plot(smooth_signal)
        #s4.plot(range(20, 200), smooth_vals)
        s4.plot(dens[10:1000])
        s4.set_yscale("log")
        plt.show()

        #plt.plot(event.eventTrace)
        #plot_dots(event.eventTrace, len(event.peptide))
        #plt.plot(smoothed)
        #plt.show()


if __name__ == "__main__":
    main()
