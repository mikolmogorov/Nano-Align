    def peptide_signal(self, peptide):
        """
        Generates theoretical signal for a given peptide
        """
        peptide = aa_to_weights(peptide)
        flanked_peptide = ("-" * (self.window - 1) + peptide +
                           "-" * (self.window - 1))
        num_peaks = len(peptide) + self.window - 1
        signal = []
        for i in xrange(0, num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            signal.append(self._svr_predict(_kmer_to_features(kmer)))
            #signal.append(_theoretical_signal(kmer))

        signal = signal / np.std(signal)
        #signal = (signal - np.mean(signal)) / np.std(signal)
        return signal

VOLUMES = {"I": 0.1688, "F": 0.2034, "V": 0.1417, "L": 0.1679,
           "W": 0.2376, "M": 0.1708, "A": 0.0915, "G": 0.0664,
           "C": 0.1056, "Y": 0.2036, "P": 0.1293, "T": 0.1221,
           "S": 0.0991, "H": 0.1673, "E": 0.1551, "N": 0.1352,
           "Q": 0.1611, "D": 0.1245, "K": 0.1713, "R": 0.2021, "-": 0.0}

theor_cache = {}
def _theoretical_signal(kmer):
    if not kmer in theor_cache:
        volumes = np.array(map(VOLUMES.get, kmer))
        unscaled = sum(volumes) / len(kmer)
        theor_cache[kmer] = unscaled
    return theor_cache[kmer]
