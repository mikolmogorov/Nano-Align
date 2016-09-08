#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Compares cross-correlation and seld-correlation of two datasets
"""

from __future__ import print_function
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance

nanoalign_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, nanoalign_root)
import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat


def correlation(mat_file_1, mat_file_2):
    """
    Draws the plot
    """

    blockades_1 = read_mat(mat_file_1)
    blockades_1 = sp._fractional_blockades(blockades_1)
    blockades_1 = sp._filter_by_duration(blockades_1, 0.5, 20)
    blockades_1 = map(lambda b: sp.discretize(sp._trim_flank_noise(b.eventTrace), 20), blockades_1)

    blockades_2 = read_mat(mat_file_2)
    blockades_2 = sp._fractional_blockades(blockades_2)
    blockades_2 = sp._filter_by_duration(blockades_2, 0.5, 20)
    blockades_2 = map(lambda b: sp.discretize(sp._trim_flank_noise(b.eventTrace), 20), blockades_2)

    self_corr = []
    cross_corr = []
    for blockade in blockades_1:
        block_self = []
        for other in blockades_1:
            block_self.append(1 - distance.correlation(blockade, other))
        block_cross = []
        for other in blockades_2:
            block_cross.append(1 - distance.correlation(blockade, other))
        self_corr.append(np.mean(block_self))
        cross_corr.append(np.mean(block_cross))

    mean_self = np.median(self_corr)
    mean_cross = np.median(cross_corr)

    matplotlib.rcParams.update({"font.size": 16})
    fig = plt.subplot()

    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.get_xaxis().tick_bottom()
    fig.get_yaxis().tick_left()
    fig.set_xlim(-0.6, 0.6)
    fig.set_ylim(-0.6, 0.6)
    fig.set_xlabel("(H3 tail, H3 tail) correlation")
    fig.set_ylabel("(H3 tail, CCL5) correlation")

    for y in [-0.4, -0.2, 0, 0.2, 0.4]:
        plt.plot((-0.6, 0.6), (y, y), "--",
                 lw=0.5, color="black")
        plt.plot((y, y), (-0.6, 0.6), "--",
                 lw=0.5, color="black")

    plt.plot((-0.6, 0.6), (mean_cross, mean_cross), "--",
             lw=1.5, color="red")
    plt.plot((mean_self, mean_self), (-0.6, 0.6), "--",
             lw=1.5, color="red")

    fig.scatter(self_corr, cross_corr, linewidth=0.5, c="dodgerblue", 
                s=30, edgecolor="blue")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("usage: cross-correlation.py blockades_file_1 blockades_file_2"
              "\n\n Plots self-correlation vs cross-correlation",
              file=sys.stderr)
        return 1

    correlation(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
