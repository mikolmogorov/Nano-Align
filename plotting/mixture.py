#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Plots blockade frequency distributions of multiple datasets
"""

from __future__ import print_function
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress, gaussian_kde

nanoalign_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, nanoalign_root)
import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat


def frequency_plot(blockade_files):
    """
    Draws the plot
    """
    datasets_names = []
    frequencies = []
    for file in blockade_files:
        blockades = read_mat(file)
        blockades = sp._fractional_blockades(blockades)
        blockades = sp._filter_by_duration(blockades, 0.5, 20)

        dataset_freqs = []
        for blockade in blockades:
            xx, yy = sp.find_peaks(blockade.eventTrace[1000:-1000])
            dataset_freqs.append(len(xx) / blockade.ms_Dwell * 5 / 4)

        frequencies.append(dataset_freqs)
        datasets_names.append(os.path.basename(file).split(".")[0])

    x_axis = np.arange(min(sum(frequencies, [])) - 10,
                       max(sum(frequencies, [])) + 10, 0.1)
    matplotlib.rcParams.update({"font.size": 16})
    fig = plt.subplot()

    colors = ["blue", "green", "red", "cyan"]
    for distr, name, color in zip(frequencies, datasets_names, colors):
        density = gaussian_kde(distr)
        density.covariance_factor = lambda: .25
        density._compute_covariance
        gauss_dens = density(x_axis)

        fig.spines["right"].set_visible(False)
        fig.spines["top"].set_visible(False)
        fig.get_xaxis().tick_bottom()
        fig.get_yaxis().tick_left()
        fig.set_ylim(0, 0.16)

        fig.plot(x_axis, gauss_dens, antialiased=True, linewidth=2, color=color,
                 alpha=0.7, label=name)
        fig.fill_between(x_axis, gauss_dens, alpha=0.5, antialiased=True,
                         color=color)

    fig.set_xlabel("Fluctuation frequency, 1/ms")
    legend = fig.legend(loc="upper left", frameon=False)
    for label in legend.get_lines():
            label.set_linewidth(3)
    for label in legend.get_texts():
        label.set_fontsize(16)
    plt.show()


def main():
    if len(sys.argv) == 1:
        print("usage: mixture.py nanospectra_file_1[ ,nanospectra_file_2...]"
              "\n\n Plots frequency distribution of multiple datasets",
              file=sys.stderr)
        return 1

    frequency_plot(sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
