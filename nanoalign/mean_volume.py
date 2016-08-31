#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Mean volume model for theoretical signal generation
"""

import numpy as np

from nanoalign.blockade_modlel import BlockadeModel


class MvBlockade(BlockadeModel):
    def __init__(self):
        super(MvBlockade, self).__init__()
        self.name = "MeanVolume"
        self.cache = {}

    def peptide_signal(self, peptide):
        """
        Generates theoretical signal for a given peptide
        """
        flanked_peptide = ("-" * (self.window - 1) + peptide +
                           "-" * (self.window - 1))
        num_peaks = len(peptide) + self.window - 1
        signal = []
        for i in xrange(0, num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            signal.append(self._signal(kmer))

        signal = (signal - np.mean(signal)) / np.std(signal)
        return signal

    def _signal(self, kmer):
        if not kmer in self.cache:
            volumes = np.array(map(self.volumes.get, kmer))
            unscaled = sum(volumes) / len(kmer)
            self.cache[kmer] = unscaled
        return self.cache[kmer]
