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
        events = read_mat(sample)
        #events = sp.filter_by_time(events, len(events[0].peptide) / 40,
        #                           len(events[0].peptide) / 5)
        sample_peaks = []
        for event in events:
            #wind = 101 / event.ms_Dwell
            #wind -= wind % 2 - 1
            #sig = savitsky_golay(event.eventTrace, wind, 3)
            sig = event.eventTrace
            xx, yy = find_peaks(sig[1000:-1000])
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


