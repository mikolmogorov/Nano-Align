#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Trains SVR model ans serializes it into a file
"""

from __future__ import print_function
import sys
import os
import argparse

import numpy as np

from nanoalign.__version__ import __version__
import nanoalign.signal_proc as sp
from nanoalign.blockade import read_mat
from nanoalign.pvalues_test import pvalues_test
from nanoalign.model_loader import store_model
from nanoalign.svr import SvrBlockade
from nanoalign.random_forest import RandomForestBlockade


def _train_random_forest(mat_files, out_file):
    """
    Trains Random Forest
    """
    peptides, signals = _get_peptides_signals(mat_files)
    model = RandomForestBlockade()
    model.train(peptides, signals)
    store_model(model, out_file)


def _train_svr(mat_files, out_file, C=1000, gamma=0.001, epsilon=0.01):
    """
    Trains SVR with the given parameters
    """
    peptides, signals = _get_peptides_signals(mat_files)
    model = SvrBlockade()
    model.train(peptides, signals, C, gamma, epsilon)
    store_model(model, out_file)


def _get_peptides_signals(mat_files):
    TRAIN_AVG = 1

    peptides = []
    signals = []
    for mat in mat_files:
        blockades = read_mat(mat)
        clusters = sp.preprocess_blockades(blockades, cluster_size=TRAIN_AVG,
                                           min_dwell=0.5, max_dwell=20)
        mat_peptide = clusters[0].blockades[0].peptide
        peptides.extend([mat_peptide] * len(clusters))

        for cluster in clusters:
            signals.append(sp.discretize(cluster.consensus, len(mat_peptide)))

    return peptides, signals


def main():
    parser = argparse.ArgumentParser(description="Nano-Align blockade model "
                                     "training", formatter_class= \
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_type", metavar="model_type",
                        choices=["svr", "rf"],
                        help="model type ('svr' or 'rf')")
    parser.add_argument("training_nanospectra", metavar="training_nanospectra",
                        help="comma-separated list of files with training "
                        "nanospectra (in mat format)")
    parser.add_argument("out_file", metavar="out_file",
                        help="path to the output file "
                        "(in Python's pickle format)")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    if args.model_type == "svr":
        _train_svr(args.training_nanospectra.split(","), args.out_file)
    else:
        _train_random_forest(args.training_nanospectra.split(","), args.out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
