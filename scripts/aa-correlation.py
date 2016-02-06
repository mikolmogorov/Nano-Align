#!/usr/bin/env python

import sys
from Bio import SeqIO
from string import maketrans
from itertools import product

WINDOW = 5
ALPHABET = "MSIL"

AA_SIZE_TRANS = maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                          "MMMMMSSSSSSIIIIIIIIILLLL-")
def aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)

def main():
    freq = {}
    for kmer in product(ALPHABET, repeat=WINDOW):
        freq["".join(kmer)] = 0

    for seq in SeqIO.parse(sys.argv[1], "fasta"):
        for i in xrange(len(seq.seq) - WINDOW + 1):
            kmer = str(seq.seq[i : i + WINDOW])
            freq[aa_to_weights(kmer)] += 1

    for kmer in sorted(freq):
        print kmer, freq[kmer]

if __name__ == "__main__":
    main()
