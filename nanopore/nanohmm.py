from __future__ import print_function
from string import maketrans
from itertools import repeat, product
from collections import namedtuple, defaultdict
import math
import random
import os

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as sio
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.decomposition import PCA

import nanopore.signal_proc as sp

ROOT_DIR = os.path.dirname(__file__)

def _signal_score(signal_1, signal_2):
    #return 1 - distance.correlation(signal_1, signal_2)
    #return spearmanr(signal_1, signal_2)[0]
    return -distance.euclidean(signal_1, signal_2)

class NanoHMM(object):
    def __init__(self, peptide_length, svr_file):
        self.window = 4
        self.svr = None
        self.trans_table = None
        self.state_to_id = {}
        self.id_to_state = {}
        self.num_peaks = peptide_length + self.window - 1
        self.alphabet = "MSIL"
        self.ext_alphabet = self.alphabet + "-"
        self.svr_means = {}

        self.set_state_space()
        self.set_init_distr()
        self.set_transition_probs(os.path.join(ROOT_DIR, "aa_freq.txt"))
        self.set_emission_probs(svr_file)
        self.svr_cache = {}

    def set_state_space(self):
        all_states = product(self.ext_alphabet, repeat=self.window)
        all_states = sorted(map("".join, all_states))
        for state_id, state in enumerate(all_states):
            self.state_to_id[state] = state_id
            self.id_to_state[state_id] = state

    def set_init_distr(self):
        self.init_distr = np.ones(len(self.state_to_id)) * 0.000001
        minisc_init = "-" * (self.window - 1) + "M"
        small_init = "-" * (self.window - 1) + "S"
        intermed_init = "-" * (self.window - 1) + "I"
        large_init = "-" * (self.window - 1) + "L"
        self.init_distr[self.state_to_id[minisc_init]] = 0.25
        self.init_distr[self.state_to_id[small_init]] = 0.25
        self.init_distr[self.state_to_id[intermed_init]] = 0.25
        self.init_distr[self.state_to_id[large_init]] = 0.25

        ending_states = [small_init[::-1], intermed_init[::-1],
                         large_init[::-1], minisc_init[::-1]]
        self.ending_states = map(self.state_to_id.get, ending_states)

    def set_transition_probs(self, prob_file):
        aa_freq = {}
        for line in open(prob_file, "r"):
            kmer, freq = line.strip().split()
            aa_freq[kmer] = int(freq)

        self.trans_table = np.ones((len(self.state_to_id),
                                    len(self.state_to_id))) * 0.000001
        for state in xrange(len(self.state_to_id)):
            kmer = self.id_to_state[state]
            if "-" not in kmer:
                total_freq = 0
                for next_aa in self.alphabet:
                    total_freq += aa_freq[kmer + next_aa]

            for next_aa in self.ext_alphabet:
                kp1mer = kmer + next_aa
                next_kmer = kp1mer[1:]
                next_state = self.state_to_id[next_kmer]
                if "-" not in kp1mer:
                    prob = float(aa_freq[kp1mer]) / total_freq
                    self.trans_table[state][next_state] = prob
                else:
                    self.trans_table[state][next_state] = \
                                        float(1) / len(self.alphabet)

    def set_emission_probs(self, svr_file):
        with open(svr_file, "r") as f:
            for line in f:
                line = line.strip()
                state, value = line.split()
                self.svr_means[state] = float(value)

    def show_fit(self, raw_signal, predicted_weights, peptide):
        fit_signal = self.peptide_signal(predicted_weights)
        theor_signal = self.peptide_signal(aa_to_weights(peptide))

        print("Experimantal score:",
              _signal_score(raw_signal, theor_signal))
        print("Fitted score:", _signal_score(fit_signal, theor_signal))

        matplotlib.rcParams.update({'font.size': 16})
        plt.plot(np.repeat(raw_signal, 2), "b-", label="experimental")
        plt.plot(np.repeat(fit_signal, 2), "g-", label="fitted")
        plt.plot(np.repeat(theor_signal, 2), "r-", label="theoretical")
        plt.xlabel("AA position")
        plt.ylabel("Normalized signal value")
        plt.legend(loc="lower right")
        plt.show()

    def show_target_vs_decoy(self, target_weights, decoy_weights, peptide):
        target_signal = self.peptide_signal(target_weights)
        decoy_signal = self.peptide_signal(aa_to_weights(decoy_weights))
        theor_signal = self.peptide_signal(aa_to_weights(peptide))

        print("Target score:",
              _signal_score(target_signal, theor_signal))
        print("Decoy score:", _signal_score(target_signal, decoy_signal))

        matplotlib.rcParams.update({'font.size': 16})
        plt.plot(target_signal, "b-", label="target", linewidth=1.5)
        plt.plot(decoy_signal, "g-", label="decoy", linewidth=1.5)
        plt.plot(theor_signal, "r-", label="theoretical", linewidth=1.5)
        plt.xlabel("AA position")
        plt.ylabel("Normalized signal value")
        plt.legend(loc="lower right")
        plt.show()

    def score(self, aa_weights_1, aa_weights_2):
        signal_1 = self.peptide_signal(aa_weights_1)
        signal_2 = self.peptide_signal(aa_weights_2)
        return _signal_score(signal_1, signal_2)

    def compute_pvalue(self, predicted_weights, peptide):
        peptide_weights = aa_to_weights(peptide)
        weights_list = list(peptide_weights)
        score = self.score(predicted_weights, peptide_weights)
        misspred = 0
        for x in xrange(10000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)

            decoy_score = self.score(decoy_weights, predicted_weights)
            if decoy_score > score:
                misspred += 1
        return float(misspred) / 10000

    def compute_pvalue_raw(self, discr_signal, peptide):
        weights_list = list(aa_to_weights(peptide))
        peptide_weights = aa_to_weights(peptide)
        theor_signal = self.peptide_signal(peptide_weights)
        misspred = 0
        score = _signal_score(discr_signal, theor_signal)
        for x in xrange(10000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)
            decoy_signal = self.peptide_signal(decoy_weights)
            decoy_score = _signal_score(discr_signal, decoy_signal)
            if decoy_score > score:
                misspred += 1
        return float(misspred) / 10000

    def emission_prob(self, state_id, observation):
        expec_mean = self.svr_means[self.id_to_state[state_id]]
        #expec_mean = _theoretical_signal(self.id_to_state[state_id])
        #return 0.000001 + norm(expec_mean, 0.01).pdf(observation)
        return math.exp(-10 * abs(observation - expec_mean))

    def peptide_signal(self, peptide):
        flanked_peptide = ("-" * (self.window - 1) + peptide +
                           "-" * (self.window - 1))
        signal = []
        for i in xrange(0, self.num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            signal.append(self.svr_means[kmer])
            #signal.append(_theoretical_signal(kmer))

        return signal

    def hmm(self, observ_seq):
        num_observ = len(observ_seq)
        num_states = len(self.init_distr)
        dp_mat = np.zeros((num_states, num_observ))
        backtrack = np.zeros((num_states, num_observ), dtype=int)

        for st in xrange(num_states):
            if self.id_to_state[st][:-1].count("-") != self.window - 1:
                dp_mat[st][0] = float("-inf")
            else:
                dp_mat[st][0] = (math.log(self.init_distr[st]) +
                                math.log(self.emission_prob(st, observ_seq[0])))
            backtrack[st][0] = -1

        #filling dp matrix
        for obs in xrange(1, num_observ):
            for st in xrange(num_states):
                #proper translocation ends
                if ("-" in self.id_to_state[st][-obs-1:num_observ - obs] or
                        self.id_to_state[st][:-obs-1].count("-") !=
                        max(0, self.window - obs - 1)):
                    dp_mat[st][obs] = float("-inf")
                    continue

                max_val = float("-inf")
                max_state = None
                for prev_st in xrange(num_states):
                    val = (dp_mat[prev_st][obs-1] +
                           math.log(self.trans_table[prev_st][st]))
                    if val > max_val:
                        max_val = val
                        max_state = prev_st
                        tt = self.trans_table[prev_st][st]

                assert dp_mat[max_state][obs-1] != float("-inf")
                dp_mat[st][obs] = (max_val +
                            math.log(self.emission_prob(st, observ_seq[obs])))
                #print(tt, self.emission_prob(st, observ_seq[obs]))
                backtrack[st][obs] = max_state

        #final state
        final_score = float("-inf")
        final_state = None
        for st in self.ending_states:
            if dp_mat[st][-1] > final_score:
                final_score = dp_mat[st][-1]
                final_state = st

        #backtracking
        states = [final_state]
        st = final_state
        for obs in xrange(num_observ - 1, 0, -1):
            st = backtrack[st][obs]
            states.append(st)

        states = states[::-1]
        return final_score, self._decode_states(states)

    def _decode_states(self, states):
        kmers = map(self.id_to_state.get, states)
        weights = map(lambda x: x[0], kmers)
        weights.extend(kmers[-1][1:])
        return "".join(weights[self.window - 1 : -self.window + 1])


def _kmer_features(kmer):
    miniscule = kmer.count("M")
    small = kmer.count("S")
    intermediate = kmer.count("I")
    large = kmer.count("L")
    return [large, intermediate, small, miniscule]


AA_SIZE_TRANS = maketrans("GASCUTDPNVBEQZHLIMKXRFYW-",
                          "MMMMMSSSSSSIIIIIIIIILLLL-")
def aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)


def _theoretical_signal(kmer):
    #VOLUMES = {"I": 0.1688, "F": 0.2034, "V": 0.1417, "L": 0.1679,
    #           "W": 0.2376, "M": 0.1708, "A": 0.0915, "G": 0.0664,
    #           "C": 0.1056, "Y": 0.2036, "P": 0.1293, "T": 0.1221,
    #           "S": 0.0991, "H": 0.1673, "E": 0.1551, "N": 0.1352,
    #           "Q": 0.1611, "D": 0.1245, "K": 0.1713, "R": 0.2021}
    VOLUMES = {"M": 0.0991, "S": 0.13225, "I": 0.1679, "L": 0.2035, "-": 0.0}

    volumes = np.array(map(VOLUMES.get, kmer))
    unscaled = sum(volumes) / len(kmer)
    return (unscaled - 0.1425) / 0.03   #magic
