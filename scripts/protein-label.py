#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Adds protein sequence record into mat file with blockades
"""

from __future__ import print_function
import sys
import os

nanoalign_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, nanoalign_root)
from nanoalign.blockade import read_mat, write_mat


def main():
    if len(sys.argv) != 3:
        print("usage: protein-label.py mat_file prot_sequence\n\n"
              "Add protein sequence record into the mat file "
              "with blockades", file=sys.stderr)
        return 1

    blockades = read_mat(sys.argv[1])
    for blockade in blockades:
        blockade.peptide = sys.argv[2]
    write_mat(blockades, sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
