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
from scipy.stats import linregress, norm
from sklearn.svm import SVR
from sklearn.decomposition import PCA

import nanopore.signal_proc as sp

ROOT_DIR = os.path.dirname(__file__)

def _signal_discordance(signal_1, signal_2):
    return -sum(map(lambda (s1, s2): (s1 - s2) ** 2,
                    zip(signal_1, signal_2)))
    #return linregress(signal_1, signal_2)[2]

class NanoHMM(object):
    def __init__(self, peptide):
        self.peptide = peptide
        self.window = 4
        self.svr = None
        self.trans_table = None
        self.state_to_id = {}
        self.id_to_state = {}
        self.num_peaks = len(self.peptide) + self.window - 1
        self.alphabet = "MSIL"
        self.ext_alphabet = self.alphabet + "-"

        self.set_state_space()
        self.set_init_distr()
        self.set_transition_probs()
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

    def set_transition_probs(self):
        aa_freq = {}
        for line in open(os.path.join(ROOT_DIR, "aa_freq.txt")):
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

    def show_fit(self, raw_signal, predicted_weights):
        fit_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(predicted_weights))
        theor_signal = map(lambda s: self.svr.predict(s)[0],
                        self.weights_to_features(aa_to_weights(self.peptide)))

        print("Experimantal score:",
              _signal_discordance(raw_signal, theor_signal))
        print("Fitted score:", _signal_discordance(fit_signal, theor_signal))

        matplotlib.rcParams.update({'font.size': 16})
        plt.plot(raw_signal, "b-", label="experimental", linewidth=1.5)
        plt.plot(fit_signal, "g-", label="fitted", linewidth=1.5)
        plt.plot(theor_signal, "r-", label="theoretical", linewidth=1.5)
        plt.xlabel("AA position")
        plt.ylabel("Normalized signal value")
        plt.legend(loc="lower right")
        plt.show()

    def show_target_vs_decoy(self, target_weights, decoy_weights):
        target_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(target_weights))
        decoy_signal = map(lambda s: self.svr.predict(s)[0],
                        self.weights_to_features(aa_to_weights(decoy_weights)))
        theor_signal = map(lambda s: self.svr.predict(s)[0],
                        self.weights_to_features(aa_to_weights(self.peptide)))

        print("Target score:",
              _signal_discordance(target_signal, theor_signal))
        print("Decoy score:", _signal_discordance(target_signal, decoy_signal))

        matplotlib.rcParams.update({'font.size': 16})
        plt.plot(target_signal, "b-", label="target", linewidth=1.5)
        plt.plot(decoy_signal, "g-", label="decoy", linewidth=1.5)
        plt.plot(theor_signal, "r-", label="theoretical", linewidth=1.5)
        plt.xlabel("AA position")
        plt.ylabel("Normalized signal value")
        plt.legend(loc="lower right")
        plt.show()

    def score(self, aa_weights_1, aa_weights_2):
        signal_1 = map(lambda s: self.svr_predict(s),
                    self.weights_to_features(aa_weights_1))
        signal_2 = map(lambda s: self.svr_predict(s),
                    self.weights_to_features(aa_weights_2))
        return _signal_discordance(signal_1, signal_2)

    def compute_pvalue(self, predicted_weights):
        peptide_weights = aa_to_weights(self.peptide)
        weights_list = list(peptide_weights)
        score = self.score(predicted_weights, peptide_weights)
        misspred = 0
        for x in xrange(1000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)

            decoy_score = self.score(decoy_weights, predicted_weights)
            if decoy_score > score:
                misspred += 1
        return float(misspred) / 1000

    def compute_pvalue_raw(self, fit_signal):
        weights_list = list(aa_to_weights(self.peptide))
        peptide_weights = aa_to_weights(self.peptide)
        theor_signal = map(lambda s: self.svr_predict(s),
                         self.weights_to_features(peptide_weights))
        misspred = 0
        score = _signal_discordance(fit_signal, theor_signal)
        for x in xrange(1000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)
            decoy_signal = map(lambda s: self.svr_predict(s),
                                self.weights_to_features(decoy_weights))
            decoy_score = _signal_discordance(theor_signal, decoy_signal)
            if decoy_score <= score:
                misspred += 1
        return float(misspred) / 1000

    def emission_prob(self, state_id, observation):
        feature = _kmer_features(self.id_to_state[state_id])
        expec_mean = self.svr_predict(feature)
        #return 0.000001 + norm(expec_mean, 0.01).pdf(observation)
        return math.exp(-100 * abs(observation - expec_mean))

    def weights_to_features(self, sequence):
        flanked_peptide = ("-" * (self.window - 1) + sequence +
                           "-" * (self.window - 1))
        features = []
        for i in xrange(0, self.num_peaks):
            kmer = flanked_peptide[i : i + self.window]
            feature = _kmer_features(kmer)

            features.append(feature)

        return features

    def learn_emissions_distr(self, events):
        features = []
        signals = []
        for event in events:
            event = sp.normalize(event)
            discretized = sp.discretize(event, self.num_peaks)
            features.extend(self.weights_to_features(aa_to_weights(self.peptide)))
            signals.extend(discretized)

        self.svr = SVR(kernel="rbf")
        self.svr.fit(features, signals)

    def svr_predict(self, feature):
        feature = tuple(feature)
        if feature not in self.svr_cache:
            self.svr_cache[feature] = self.svr.predict(feature)[0]
        return self.svr_cache[feature]

    def score_svm(self, events):
        def rand_jitter(arr):
            stdev = .01*(max(arr)-min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        features = []
        signals = []
        for event in events:
            event = sp.normalize(event)
            discretized = sp.discretize(event, self.num_peaks)
            features.extend(self.weights_to_features(aa_to_weights(self.peptide)))
            signals.extend(discretized)

        print(self.svr.score(features, signals))
        #pca = PCA(2)
        #pca.fit(features)
        #new_x = pca.transform(features)
        #plt.hist(signals, bins=50)
        #plt.show()
        #plt.scatter(rand_jitter(new_x[:, 0]), rand_jitter(new_x[:, 1]), s=50,
        #            c=signals, alpha=0.5)
        #plt.show()

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
