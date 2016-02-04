#!/usr/bin/env python


from __future__ import print_function

import sys
import math
from collections import defaultdict

import numpy as np
import nanopore.signal_proc as sp
import matplotlib.pyplot as plt
import matplotlib
import scipy.fftpack as fftpack
from scipy.stats import linregress, gaussian_kde
from scipy.signal import find_peaks_cwt, periodogram
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn import mixture
from scipy import signal
import scipy.io as sio


def peak_features(signal, n_peaks, minimum=False, ranged=False):
    signal = np.array(signal)
    WINDOW = 10

    peaks = []
    for pos in xrange(WINDOW, len(signal) - WINDOW):
        left = signal[pos - WINDOW: pos] - signal[pos]
        right = signal[pos + 1: pos + WINDOW + 1] - signal[pos]

        if not minimum:
            if (left < 0).all() and (right < 0).all():
                peaks.append((pos, abs(np.mean(left) + np.mean(right))))
        else:
            if (left > 0).all() and (right > 0).all():
                peaks.append((pos, signal[pos]))

    selected = sorted(peaks, key=lambda p: p[1], reverse=not minimum)[:n_peaks]
    xx = map(lambda p: p[0], selected)
    if not ranged:
        xx.sort()
    yy = map(lambda p: signal[p], xx)
    return xx, yy


def get_peak_spectra(signal):
    def peaks_score(deriv, n_dots, start, end):
        peaks = np.linspace(start, end, n_dots)
        deriv_vals = map(lambda x: deriv[x], peaks)
        return np.std(np.array(deriv_vals) ** 2)

    xx, yy = peak_features(signal, 2000)
    der = np.gradient(signal, 2)

    vals = []
    r = range(20, 200)
    for i in r:
        vals.append(peaks_score(der, i, xx[0], xx[-1]))
    vals = (vals - min(vals)) / (max(vals) - min(vals))
    #return vals
    return savitsky_golay(vals, 51, 3)


def characteristic_peaks(signal):
    peaks, yy = peak_features(get_spectra(signal), 2000)
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


def draw_plot():
    h32 = "../reversed/H32.mat"
    h4 = "../reversed/H4_all.mat"
    ccl5 = "../reversed/CCL5.mat"
    h33 = "../reversed/H3N.mat"

    d1 = "../datasets/ZD349_H4_D3.mat"
    d2 = "../datasets/ZD349_H4_D4.mat"
    d3 = "../datasets/ZD349_H4_D5.mat"

    peaks = []
    for sample in [h32, h4, ccl5, h33]:
    #for sample in [d1, d2, d3]:
        print(sample)
        events = sp.read_mat(sample)
        #events = sp.filter_by_time(events, len(events[0].peptide) / 40,
        #                           len(events[0].peptide) / 5)
        sample_peaks = []
        for event in events:
            #wind = 101 / event.ms_Dwell
            #wind -= wind % 2 - 1
            #sig = savitsky_golay(event.eventTrace, wind, 3)
            sig = event.eventTrace
            xx, yy = peak_features(sig[1000:-1000], 2000)
            sample_peaks.append(len(xx) / event.ms_Dwell * 5 / 4)

        peaks.append(sample_peaks)

    #plt.hist(peaks, bins=20, histtype="bar", normed=1,
    #         label=["H32", "H4", "CCL5", "H3"])
    #plt.hist(peaks, bins=20, histtype="bar", normed=1,
    #         label=["Day 3", "Day 4", "Day 5"])
    #plt.hist(peaks, bins=100)
    #plt.xlabel("Noise frequency, 1/msec")
    #plt.legend()
    #plt.show()

    x_axis = np.arange(0, 50, 0.1)
    matplotlib.rcParams.update({"font.size": 16})
    fig = plt.subplot()

    colors = ["blue", "green", "red", "cyan"]
    labels = ["H3.2", "H4", "CCL5", "H3N"]

    for i, distr in enumerate(peaks):
        density = gaussian_kde(distr)
        density.covariance_factor = lambda: .25
        density._compute_covariance
        gauss_dens = density(x_axis)

        fig.spines["right"].set_visible(False)
        fig.spines["top"].set_visible(False)
        fig.get_xaxis().tick_bottom()
        fig.get_yaxis().tick_left()
        fig.set_ylim(0, 0.16)

        fig.plot(x_axis, gauss_dens, antialiased=True, linewidth=2, color=colors[i],
                 alpha=0.7, label=labels[i])
        fig.fill_between(x_axis, gauss_dens, alpha=0.5, antialiased=True,
                         color=colors[i])
        #fig.hist(distr, normed=1, range=(0,61), bins=20)

    fig.set_xlabel("Fluctuation frequency, 1/ms")
    legend = fig.legend(loc="lower left", frameon=False)
    for label in legend.get_lines():
            label.set_linewidth(3)
    for label in legend.get_texts():
        label.set_fontsize(16)
    plt.show()

    """
    s1.hist(peaks, bins=20, histtype="bar", normed=1,
             label=["H32", "H4", "CCL5"])
    s1.set_xlabel("Noise frequency, 1/msec")
    s1.legend()

    all_peaks = sum(peaks, [])
    d_hist, bin_edges = np.histogram(all_peaks, bins=200)
    s2.plot(bin_edges[1:], d_hist)

    window = signal.gaussian(5, std=1)
    smooth = np.convolve(d_hist, window, mode="same")
    s3.plot(bin_edges[1:], smooth)

    plt.show()
    """


def test_noise(filename, real_prot):
    signal = sio.loadmat(filename)["b"].squeeze()
    real_prot = sp.read_mat(real_prot)

    #dwell = 5
    #vals = []
    #for i in xrange(0, len(signal) / length):
    #    chunk = signal[i * length :  (i + 1) * length]
    #    xx, yy = peak_features(chunk, 10000)
    #    vals.append(len(xx) / dwell)

    for e in real_prot:
        #plt.plot(e.eventTrace)
        #plt.show()
        #stds.append(np.std(e.eventTrace))
    #plt.hist(stds, bins=50)
    #plt.show()
    #print(np.mean(stds))
    #print(np.std(signal))

        dwell = e.ms_Dwell
        length = int(dwell * 100)
        print(dwell, np.std(e.eventTrace), np.std(signal[:length]))

        plt.plot(np.linspace(0, 10000, length),
                 signal[:length] - np.median(signal[:length]), label="noise")
        plt.plot(e.eventTrace - np.median(e.eventTrace),
                 label="signal")
        plt.legend()
        plt.show()

    print(len(vals))
    plt.hist(vals, bins=50)
    plt.show()


def pps_dist(events):
    peaks_count = {}
    a = []
    b = []
    for event in events:
        #wind = 101 / event.ms_Dwell
        #wind -= wind % 2 - 1
        #div = savitsky_golay(np.gradient(event.eventTrace), wind, 3)
        #xx, yy = peak_features(div[1000:-1000], 2000)
        #signal = savitsky_golay(event.eventTrace, wind, 3)
        signal = event.eventTrace[1000:-1000]
        xx, yy = peak_features(signal, 2000)
        peaks_count[event] = len(xx) / event.ms_Dwell * 5 / 4

        b.append(len(xx))
        a.append(event.ms_Dwell * 4 / 5)

    print(linregress(a, b)[:2])
    plt.scatter(a, b)
    plt.show()

    #plt.scatter(map(lambda e: np.mean(e.eventTrace), events),
    #            map(peaks_count.get, events))
    #plt.show()

    mean = np.mean(peaks_count.values())
    errors = map(lambda e: peaks_count[e] - mean, events)
    lengths = map(lambda e: e.ms_Dwell, events)

    plt.scatter(lengths, errors)
    plt.show()

    print(mean, np.std(peaks_count.values()))
    plt.hist(peaks_count.values(), bins=100)
    plt.show()


def analyse(events):
    all_peaks = []

    for event in events:
        xx, yy = peak_features(event.eventTrace, 2000)
        diff_1 = sorted(np.array(xx)[1:] - np.array(xx)[:-1])
        diff_2 = sorted(np.array(xx)[2:] - np.array(xx)[:-2])
        diff_3 = sorted(np.array(xx)[3:] - np.array(xx)[:-3])
        diff = diff_1 + diff_2 + diff_3
        diff = np.array(diff) * event.ms_Dwell

        true_peaks = np.linspace(xx[0], xx[-1], len(event.peptide))
        #print(len(event.peptide))

        #peak_spectra = get_peak_spectra(event.eventTrace)

        #smoothing with Gaussian window
        d_hist, bin_edges = np.histogram(diff, bins=100, range=(100, 1000))
        window = signal.gaussian(21, std=1)
        smooth = np.convolve(d_hist, window, mode="same")
        smooth_smooth = savitsky_golay(smooth, 5, 3)

        hist_x, hist_y = peak_features(smooth_smooth, 10)
        char_peaks = map(lambda x: int(bin_edges[x + 1]), hist_x)

        cd = np.mean(np.array(char_peaks[1:]) - np.array(char_peaks[:-1]))
        print(event.ms_Dwell, 10000 / len(xx) * event.ms_Dwell, char_peaks, cd)
        gcd_plot, gcds = gcd_fuzz(char_peaks)
        print("gcd", gcds)

        fft = fftpack.rfft(event.eventTrace)
        #length = gcds[0]

        f, (s1, s2, s3, s4) = plt.subplots(4)
        s1.plot(event.eventTrace)
        s1.scatter(xx, yy)
        #s1.scatter(true_peaks, map(lambda x: event.eventTrace[x], true_peaks))

        #s2.plot(bin_edges[1:], smooth_smooth)
        grad = savitsky_golay(np.gradient(event.eventTrace), 101, 3)
        g_x, g_y = peak_features(grad, 2000)
        s2.plot(grad)
        s2.scatter(g_x, g_y)
        #s3.plot(range(20, len(peak_spectra) + 20), peak_spectra)
        #s5.plot(range(50, 300), gcds)
        #s4.plot(bin_edges[1:], smooth_smooth)
        s3.plot(gcd_plot)
        s4.plot(np.linspace(0, 1000 / event.ms_Dwell, 1000),
                savitsky_golay(fft ** 2, 31, 3)[:1000])
        s4.set_yscale("log")
        plt.show()

        if gcds:
            all_peaks.append(gcds[0])
        #if not math.isnan(cd):
        #    all_peaks.append(10000 / cd / event.ms_Dwell)

    print(np.mean(all_peaks), np.std(all_peaks))
    plt.hist(all_peaks, bins=100)
    plt.show()


def main():
    #test_noise(sys.argv[1], sys.argv[2])
    #draw_plot()
    events = sp.read_mat(sys.argv[1])
    events = sp.filter_by_time(events, 0.5, 20.0)
    sp.normalize(events)

    pps_dist(events)
    analyse(events)

    #old_analyse(events)
    #cross_corr(events)
    #score_distr(events)


if __name__ == "__main__":
    main()
