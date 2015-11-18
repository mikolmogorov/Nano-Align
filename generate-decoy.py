#!/usr/bin/env python

import random
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

#CCL5
#PROT = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
#CXCL1
#PROT = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
#H3N
PROT = "ARTKQTARKSTGGKAPRKQL"[::-1]
DB_SIZE = 1000

SeqIO.write(SeqRecord(seq=Seq(PROT), id="target", description=""),
            sys.stdout, "fasta")
prot_list = list(PROT)
for i in xrange(DB_SIZE):
    random.shuffle(prot_list)
    SeqIO.write(SeqRecord(seq=Seq("".join(prot_list)), id="decoy_{0}".format(i),
                                  description=""), sys.stdout, "fasta")
