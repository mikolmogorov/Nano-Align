#!/usr/bin/env python

from __future__ import print_function
import sys
from string import maketrans
from itertools import repeat, product
from collections import namedtuple, defaultdict
import math
import random

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import linregress, norm
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from nanopore import get_data, theoretical_signal, get_averages


class NanoHMM(object):
    def __init__(self):
        self.peptide = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
        #self.peptide = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
        #self.peptide = "ARTKQTARKSTGGKAPRKQL"
        self.window = 4
        self.reverse = True
        self.svr = None
        self.trans_table = None
        self.state_to_id = {}
        self.id_to_state = {}
        self.num_peaks = len(self.peptide) + self.window - 1

        self.set_state_space()
        self.set_init_distr()
        self.set_trans_table()

    def set_state_space(self):
        all_acids = "-TLMH"
        all_states = product(all_acids, repeat=self.window)
        all_states = sorted(map("".join, all_states))
        for state_id, state in enumerate(all_states):
            self.state_to_id[state] = state_id
            self.id_to_state[state_id] = state

    def set_init_distr(self):
        self.init_distr = np.ones(len(self.state_to_id)) * 0.000001
        tiny_init = "-" * (self.window - 1) + "T"
        light_init = "-" * (self.window - 1) + "L"
        med_init = "-" * (self.window - 1) + "M"
        heavy_init = "-" * (self.window - 1) + "H"
        self.init_distr[self.state_to_id[light_init]] = 0.25
        self.init_distr[self.state_to_id[med_init]] = 0.25
        self.init_distr[self.state_to_id[heavy_init]] = 0.25
        self.init_distr[self.state_to_id[tiny_init]] = 0.25

        ending_states = [light_init[::-1], med_init[::-1], heavy_init[::-1], tiny_init[::-1]]
        self.ending_states = map(self.state_to_id.get, ending_states)

    def show_fit(self, raw_signal, predicted_weights):
        fit_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(predicted_weights))
        theor_signal = map(lambda s: self.svr.predict(s)[0],
                        self.weights_to_features(_aa_to_weights(self.peptide)))
        plt.plot(raw_signal, "bx-", label="raw signal")
        plt.plot(fit_signal, "gx-", label="fit signal")
        plt.plot(theor_signal, "rx-", label="theory")
        plt.legend(loc="lower right")
        plt.show()

    def compute_pvalue(self, predicted_weights):
        fit_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(predicted_weights))
        peptide_weights = _aa_to_weights(self.peptide)
        theor_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(peptide_weights))
        misspred = 0
        score = _signal_discordance(fit_signal, theor_signal)
        #print("True score:", score)
        weights_list = list(peptide_weights)
        for x in xrange(1000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)
            decoy_signal = map(lambda s: self.svr.predict(s)[0],
                                self.weights_to_features(decoy_weights))
            decoy_score = _signal_discordance(theor_signal, decoy_signal)
            #print("Decoy score:", decoy_score)
            if decoy_score <= score:
                misspred += 1
        return float(misspred) / 1000

    def compute_pvalue_raw(self, fit_signal):
        weights_list = list(_aa_to_weights(self.peptide))
        peptide_weights = _aa_to_weights(self.peptide)
        theor_signal = map(lambda s: self.svr.predict(s)[0],
                         self.weights_to_features(peptide_weights))
        misspred = 0
        score = _signal_discordance(fit_signal, theor_signal)
        #print("True score:", score)
        for x in xrange(1000):
            random.shuffle(weights_list)
            decoy_weights = "".join(weights_list)
            decoy_signal = map(lambda s: self.svr.predict(s)[0],
                                self.weights_to_features(decoy_weights))
            decoy_score = _signal_discordance(theor_signal, decoy_signal)
            #print("Decoy score:", decoy_score)
            if decoy_score <= score:
                misspred += 1
        return float(misspred) / 1000

    def set_trans_table(self):
        self.trans_table = np.ones((len(self.state_to_id),
                                    len(self.state_to_id))) * 0.000001
        for st_1, st_2 in product(xrange(len(self.state_to_id)), repeat=2):
            seq_1 = self.id_to_state[st_1]
            seq_2 = self.id_to_state[st_2]
            if seq_1[1:] == seq_2[:-1]:
                self.trans_table[st_1][st_2] = 0.20

    def emission_prob(self, state_id, observation):
        feature = _kmer_features(self.id_to_state[state_id])
        expec_mean = self.svr.predict(feature)[0]
        x = norm(expec_mean, 0.1).pdf(observation)
        return x

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
            event = _normalize(event)
            features.extend(self.weights_to_features(_aa_to_weights(self.peptide)))
            discretized = _discretize(event, self.num_peaks)
            signals.extend(discretized)

        #self.show_pca(features, signals)
        self.svr = SVR()
        self.svr.fit(features, signals)
        print(self.svr.score(features, signals))

    def show_pca(self, features, signals):
        def rand_jitter(arr):
            stdev = .01*(max(arr)-min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        pca = PCA(2)
        pca.fit(features)
        new_x = pca.transform(features)
        plt.hist(signals, bins=50)
        plt.show()
        plt.scatter(rand_jitter(new_x[:, 0]), rand_jitter(new_x[:, 1]), s=50,
                    c=signals, alpha=0.5)
        plt.show()

    def hmm(self, observ_seq):
        num_observ = len(observ_seq)
        num_states = len(self.init_distr)
        dp_mat = np.zeros((num_states, num_observ))
        backtrack = np.zeros((num_states, num_observ), dtype=int)

        for st in xrange(num_states):
            dp_mat[st][0] = (math.log(self.init_distr[st]) +
                             math.log(self.emission_prob(st, observ_seq[0])))
            backtrack[st][0] = -1

        #filling dp matrix
        for obs in xrange(1, num_observ):
            for st in xrange(num_states):
                if (self.id_to_state[st].count("-") and
                    self.window <= obs < num_observ - self.window):
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

                dp_mat[st][obs] = (max_val +
                            math.log(self.emission_prob(st, observ_seq[obs])))
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


def _normalize(signal):
    median = np.median(signal)
    return signal / median


def _discretize(signal, num_peaks):
    discrete = []
    peak_shift = len(signal) / (num_peaks - 1)
    for i in xrange(0, num_peaks):
        signal_pos = i * peak_shift
        left = max(0, signal_pos - peak_shift / 2)
        right = min(len(signal), signal_pos + peak_shift / 2)
        discrete.append(np.mean(signal[left:right]))

    return discrete


AA_SIZE_TRANS = maketrans("GASCTDPNVEQHLIMKRFYW-",
                          "TTTTLLLLLMMMMMMMHHHH-")
def _aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)


def _signal_discordance(signal_1, signal_2):
    #result = 0.0
    #for a, b in zip(signal_1, signal_2):
    #    result += (a - b) ** 2
    regress = linregress(signal_1, signal_2)
    #print(regress)
    return 1 - regress[2]


def _hamming_dist(str_1, str_2):
    res = 0
    for a, b in zip(str_1, str_2):
        res += int(a != b)
    return res


def _kmer_features(kmer):
    tiny = kmer.count("T")
    light = kmer.count("L")
    medium = kmer.count("M")
    heavy = kmer.count("H")
    return [heavy, medium, light, tiny]


def _most_common(lst):
    return max(set(lst), key=lst.count)


TRAIN_AVG = 20
TEST_AVG = 3
FLANK = 10


def main():
    events = get_data(sys.argv[1])
    nano_hmm = NanoHMM()
    train_events = get_averages(events, TRAIN_AVG, FLANK,
                                nano_hmm.reverse)
    test_events = get_averages(events, TEST_AVG, FLANK,
                               nano_hmm.reverse)

    nano_hmm.learn_emissions_distr(train_events)
    correct_weights = _aa_to_weights(nano_hmm.peptide)
    print(correct_weights, "\n")
    profile = [[] for x in xrange(len(correct_weights))]

    identified = 0
    identified_raw = 0
    print("Sequence\t\tHMM_score\tFrac_corr\tFit_pvalue\tRaw_pvalue")
    for event in test_events:
        event = _normalize(event)
        event = _discretize(event, nano_hmm.num_peaks)
        score, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        p_value_raw = nano_hmm.compute_pvalue_raw(event)
        if p_value < 0.1:
            identified += 1
        if p_value_raw < 0.1:
            identified_raw += 1

        accuracy = _hamming_dist(weights, correct_weights)
        accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
        print("{0}\t{1:5.2f}\t{2}\t{3}\t{4}".format(weights, score, accuracy,
                                                    p_value, p_value_raw))
        for pos, aa in enumerate(weights):
            profile[pos].append(aa)

        #nano_hmm.show_fit(event, weights)

    profile = "".join(map(_most_common, profile))
    accuracy = _hamming_dist(profile, correct_weights)
    accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
    print()
    print(profile, accuracy)
    print("Identified:", float(identified) / len(test_events))
    print("Identified raw:", float(identified_raw) / len(test_events))


if __name__ == "__main__":
    main()
