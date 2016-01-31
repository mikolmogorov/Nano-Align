from __future__ import print_function
from string import maketrans
from itertools import repeat, product
from collections import namedtuple, defaultdict
import math
import random
import os
import pickle

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as sio
from scipy.spatial import distance
from scipy.stats import spearmanr, norm
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

import nanopore.signal_proc as sp

ROOT_DIR = os.path.dirname(__file__)

def _signal_score(signal_1, signal_2):
    #return 1 - distance.correlation(signal_1, signal_2)
    #return spearmanr(signal_1, signal_2)[0]
    #return -np.median(np.array(signal_1) - np.array(signal_2))
    return -distance.sqeuclidean(signal_1, signal_2)

class NanoHMM(object):
    def __init__(self, peptide_length, svr_file):
        self.window = 4
        self.svr = None
        self.trans_table = None
        self.state_to_id = {}
        self.id_to_state = {}
        self.alphabet = "MSIL"
        self.ext_alphabet = self.alphabet + "-"

        self.svr = None
        self.svr_cache = {}

        self.set_emission_probs(svr_file)
        self.svr_cache = {}

    def svr_predict(self, feature):
        if feature not in self.svr_cache:
            np_feature = np.array(feature).reshape(1, -1)
            self.svr_cache[feature] = self.svr.predict(np_feature)[0]
        return self.svr_cache[feature]

    def set_emission_probs(self, svr_file):
        self.svr = pickle.load(open(svr_file, "rb"))

    def signal_peptide_score(self, signal, peptide):
        theor_signal = self.peptide_signal(peptide)
        return _signal_score(signal, theor_signal)

    def get_errors(self, peptide, exp_signal):
        theor_signal = self.peptide_signal(peptide)
        flanked_peptide = ("-" * (self.window - 1) + peptide +
                           "-" * (self.window - 1))
        num_peaks = len(peptide) + self.window - 1

        errors = []
        for i in xrange(0, num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            if "-" not in kmer:
                for aa in kmer:
                    errors.append((_theoretical_signal(aa),
                               exp_signal[i] - theor_signal[i]))
        return errors

    def compute_pvalue_raw(self, discr_signal, peptide):
        weights_list = list(peptide)
        misspred = 0
        score = self.signal_peptide_score(discr_signal, peptide)
        decoy_winner = None
        #scores = []
        for x in xrange(10000):
            random.shuffle(weights_list)
            #weights_list = [random.choice(AAS) for _ in xrange(len(peptide))]
            decoy_weights = "".join(weights_list)
            decoy_signal = self.peptide_signal(decoy_weights)
            decoy_score = self.signal_peptide_score(discr_signal, decoy_weights)
            #decoy_score = self.signal_peptide_score(decoy_signal, peptide)
            #scores.append(decoy_score)
            if decoy_score > score:
                decoy_winner = decoy_signal
                misspred += 1

        p_value = float(misspred) / 10000
        #print(np.median(scores), score)
        #print(p_value)
        #self.plot_raw_vs_theory(discr_signal, peptide, decoy_winner)
        return p_value

    def plot_raw_vs_theory(self, discr_signal, peptide, decoy_winner):
        theor_signal = self.peptide_signal(peptide)

        print("Score:", _signal_score(discr_signal, theor_signal))
        if decoy_winner is not None:
            print("Decoy score:", _signal_score(discr_signal, decoy_winner))

        plt.plot(np.repeat(discr_signal, 2), "b-", label="experimental")
        plt.plot(np.repeat(theor_signal, 2), "r-", label="theory")
        if decoy_winner is not None:
            plt.plot(np.repeat(decoy_winner, 2), "g-", label="decoy")
        plt.xlabel("Sampling points")
        plt.ylabel("Normalized signal")
        plt.legend(loc="upper right")
        plt.show()

    def peptide_signal(self, peptide):
        peptide = aa_to_weights(peptide)
        flanked_peptide = ("-" * (self.window - 1) + peptide +
                           "-" * (self.window - 1))
        num_peaks = len(peptide) + self.window - 1
        signal = []
        for i in xrange(0, num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            signal.append(self.svr_predict(_kmer_features(kmer)))
            #signal.append(_theoretical_signal(kmer))

        signal = signal / np.std(signal)
        #signal = (signal - np.mean(signal)) / np.std(signal)
        return signal


def _kmer_features(kmer):
    miniscule = kmer.count("M")
    small = kmer.count("S")
    intermediate = kmer.count("I")
    large = kmer.count("L")
    return (large, intermediate, small, miniscule)


AAS = "GASCUTDPNVBEQZHLIMKXRFYW"
AA_SIZE_TRANS = maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                          "MMMMMSSSSSSIIIIIIIIILLLL-")
def aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)


#VOLUMES = {"M": 0.0991, "S": 0.13225, "I": 0.1679, "L": 0.2035, "-": 0.0}
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
