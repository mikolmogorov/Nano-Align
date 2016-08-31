#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
SVR model for theoretical signal generation
"""

from __future__ import print_function
from string import maketrans

import numpy as np
from sklearn.svm import SVR

from nanoalign.blockade_modlel import BlockadeModel


class SvrBlockade(BlockadeModel):
    def __init__(self):
        super(SvrBlockade, self).__init__()
        self.name = "SVR"
        self.svr_cache = {}

    def _svr_predict(self, feature_vec):
        """
        Predicts signal for a feature vector
        """
        if feature_vec not in self.svr_cache:
            np_feature = np.array(feature_vec).reshape(1, -1)
            self.svr_cache[feature_vec] = self.predictor.predict(np_feature)[0]
        return self.svr_cache[feature_vec]

    def train(self, peptides, signals, C=1000, gamma=0.001, epsilon=0.01):
        """
        Trains SVR model
        """
        self.predictor = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        features = map(lambda p: self._peptide_to_features(p), peptides)
        train_features = np.array(sum(features, []))
        train_signals = np.array(sum(signals, []))
        assert len(train_features) == len(train_signals)

        self.predictor.fit(train_features, train_signals)
        print(self.predictor.score(train_features, train_signals))

    def peptide_signal(self, peptide):
        """
        Generates theoretical signal for a given peptide
        """
        assert self.predictor is not None

        features = self._peptide_to_features(peptide)
        signal = np.array(map(lambda x: self._svr_predict(x), features))
        #normalize the signal's amplitude
        signal = signal / np.std(signal)
        return signal

    def _peptide_to_features(self, peptide):
        """
        Converts peptide into a list of feature vectors
        """
        aa_weights = _aa_to_weights(peptide)
        num_peaks = len(aa_weights) + self.window - 1
        flanked_peptide = ("-" * (self.window - 1) + aa_weights +
                           "-" * (self.window - 1))
        features = []
        for i in xrange(0, num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            feature = _kmer_to_features(kmer)
            features.append(feature)

        return features


def _kmer_to_features(kmer):
    """
    Converts kmer in reduced alphabet into a feature vector
    """
    miniscule = kmer.count("M")
    small = kmer.count("S")
    intermediate = kmer.count("I")
    large = kmer.count("L")
    return (large, intermediate, small, miniscule)


AA_SIZE_TRANS = maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                          "MMMMMSSSSSSIIIIIIIIILLLL-")
def _aa_to_weights(kmer):
    """
    Converts AAs into the reduced alphabet
    """
    return kmer.translate(AA_SIZE_TRANS)
