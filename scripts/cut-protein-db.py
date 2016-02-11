#!/usr/bin/env python2.7

#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Filters proteins of chosen length
"""

from __future__ import print_function
from Bio import SeqIO
import sys


def main():
    if len(sys.argv) != 4:
        print("usage: cut-proteins-db.py fasta_db min_len max_len\n\n"
              "Filters proteins of chosen length from a FASTA database",
              file=sys.stderr)
        return 1

    fasta_db = sys.argv[1]
    min_len = int(sys.argv[2])
    max_len = int(sys.argv[3])
    SeqIO.write(filter(lambda s: min_len <= len(s.seq) <= max_len,
                       SeqIO.parse(fasta_db, "fasta")), sys.stdout, "fasta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
