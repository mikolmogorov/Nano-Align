#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Flips blockades signals according to the protein's AA order
"""

from __future__ import print_function
import sys
import os

nanoalign_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, nanoalign_root)
from nanoalign.identifier import Identifier
from nanoalign.blockade import read_mat, write_mat
import nanoalign.signal_proc as sp
from nanoalign.model_loader import load_model


def flip(blockades, model_file):
    """
    Flips blockades
    """
    blockade_model = load_model(model_file)
    identifier = Identifier(blockade_model)

    peptide = blockades[0].peptide
    clusters = sp.preprocess_blockades(blockades, cluster_size=1,
                                       min_dwell=0.0, max_dwell=1000)

    print("Num\tFwd_dst\tRev_dst\t\tNeeds_flip", file=sys.stderr)

    num_reversed = 0
    new_blockades = []
    for num, cluster in enumerate(clusters):
        discr_signal = sp.discretize(cluster.consensus, len(peptide))

        fwd_dist = identifier.signal_protein_distance(discr_signal, peptide)
        rev_dist = identifier.signal_protein_distance(discr_signal,
                                                      peptide[::-1])
        print("{0}\t{1:5.2f}\t{2:5.2f}\t\t{3}"
                .format(num + 1, fwd_dist, rev_dist, fwd_dist > rev_dist),
                file=sys.stderr)

        new_blockades.append(cluster.blockades[0])
        if fwd_dist > rev_dist:
            new_blockades[-1].eventTrace = new_blockades[-1].eventTrace[::-1]
            num_reversed += 1

    print("Reversed:", num_reversed, "of", len(blockades), file=sys.stderr)
    return new_blockades


def main():
    if len(sys.argv) != 4:
        print("usage: flip-blockades.py blockades_in model_file flipped_out\n\n"
              "Orients blockade signals according to the AA order "
              "in the protein of origin")
        return 1

    blockades_in = sys.argv[1]
    blockades_out = sys.argv[3]
    svr_file = sys.argv[2]

    blockades = read_mat(blockades_in)
    rev_blockades = flip(blockades, svr_file)
    write_mat(rev_blockades, blockades_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
