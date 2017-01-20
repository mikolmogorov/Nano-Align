#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Plots identification p-values as a function of number blockades in
consensus
"""

from __future__ import print_function
import os
import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

nanoalign_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, nanoalign_root)
from nanoalign.pvalues_test import pvalues_test
from nanoalign.model_loader import load_model
#from nanoalign.svr_blockade import SvrBlockade


def full_identify(blockades_file, model_file, db_file):
    """
    Computes pvalues
    """
    blockade_model = load_model(model_file)
    #svr_model = SvrBlockade()
    #svr_model.load_from_pickle(svr_file)

    boxes = []
    for avg in xrange(1, 21):
        p_values = []
        for _ in xrange(avg):
            p_value, rank = pvalues_test(blockades_file, avg, blockade_model, db_file,
                                         False, open(os.devnull, "w"))
            p_values.append(p_value)

        boxes.append(p_values)
        print(avg, np.median(p_values), file=sys.stderr)

    plot_pvalues(boxes)


def plot_pvalues(pvalues):
    """
    Draws the plot
    """
    matplotlib.rcParams.update({"font.size": 16})
    matplotlib.rcParams["ytick.major.size"] = 10
    matplotlib.rcParams["ytick.major.width"] = 2
    matplotlib.rcParams["ytick.minor.size"] = 6
    matplotlib.rcParams["ytick.minor.width"] = 1
    matplotlib.rcParams["xtick.major.size"] = 10
    matplotlib.rcParams["xtick.major.width"] = 2
    matplotlib.rcParams["xtick.minor.size"] = 6
    matplotlib.rcParams["xtick.minor.width"] = 1

    fig = plt.subplot()

    x_axis = range(1, len(pvalues) + 1)
    pvalues_medians = map(np.median, pvalues)

    fig.errorbar(x_axis, pvalues_medians, fmt="o-",
                 label="p-values", linewidth=1.5)

    for y in [0.1, 0.01, 0.001]:
        plt.plot(x_axis, [y] * len(x_axis), "--",
                 lw=0.5, color="black", alpha=0.3)

    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.get_xaxis().tick_bottom()
    fig.get_yaxis().tick_left()
    fig.set_ylim(0.001, 1)

    fig.set_xlabel("Consensus size")
    fig.set_ylabel("Median p-value")
    fig.set_yscale("log", nonposy="clip")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plots identification "
                     "p-values as a function of the number of nanospectra "
                     "in a cluster", formatter_class= \
                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("nanospectra_file", metavar="nanospectra_file",
                        help="input file with nanospectra (in mat format)")
    parser.add_argument("model_file", metavar="model_file",
                        help="path to the trained model file "
                        "('-' for MV model)")
    parser.add_argument("-d", "--database", dest="database",
                        metavar="database", help="database file (in FASTA "
                        "format). If not set, random database is generated",
                        default=None)
    args = parser.parse_args()
    full_identify(args.nanospectra_file, args.model_file, args.database)

    return 0


if __name__ == "__main__":
    sys.exit(main())
