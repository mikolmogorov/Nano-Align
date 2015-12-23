#!/usr/bin/env python

from Bio import SeqIO
import sys

SeqIO.write(filter(lambda s: int(sys.argv[2]) <= len(s.seq) <= int(sys.argv[3]),
                   SeqIO.parse(sys.argv[1], "fasta")), sys.stdout, "fasta")
