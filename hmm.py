#!/usr/bin/env python

from __future__ import print_function
import sys
from string import maketrans
from itertools import repeat, product
from collections import namedtuple

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import linregress, norm

from nanopore import get_data, theoretical_signal, get_averages


VolSigRegression = namedtuple("VolSigRegression", ["slope", "intercept", "std"])


class NanoHMM(object):
    def __init__(self):
        self.peptide = "ARTKQTARKSTGGKAPRKQL"
        self.window = 4
        self.average = 30
        self.flank = 50
        self.reverse = True
        self.regression = None
        self.init_distr = None
        self.trans_table = None
        self.state_to_id = {}
        self.id_to_state = {}

        self.set_state_space()
        self.set_init_distr()
        self.set_trans_table()

    def set_state_space(self):
        all_acids = "-LMH"
        all_states = product(all_acids, repeat=self.window)
        all_states = sorted(map("".join, all_states))
        for state_id, state in enumerate(all_states):
            self.state_to_id[state] = state_id
            self.id_to_state[state_id] = state

    def set_init_distr(self):
        self.init_distr = np.zeros(len(self.state_to_id))
        light_init = "-" * (self.window - 1) + "L"
        med_init = "-" * (self.window - 1) + "M"
        heavy_init = "-" * (self.window - 1) + "H"
        self.init_distr[self.state_to_id[light_init]] = float(1) / 3
        self.init_distr[self.state_to_id[med_init]] = float(1) / 3
        self.init_distr[self.state_to_id[heavy_init]] = float(1) / 3

    def set_trans_table(self):
        self.trans_table = np.zeros((len(self.state_to_id),
                                     len(self.state_to_id)))
        for st_1, st_2 in product(xrange(len(self.state_to_id)), repeat=2):
            seq_1 = self.id_to_state[st_1]
            seq_2 = self.id_to_state[st_2]
            if seq_1[1:] == seq_2[:-1]:
                self.trans_table[st_1][st_2] = 0.25

    def emission_prob(state_id, observation):
        volume = _avg_volume(self.id_to_state(state_id))
        expec_mean = self.regression.intercept + self.regression.slope * volume
        return norm(expec_mean, self.regression.std).cdf(observation)

    def learn_emissions_distr(self, events):
        volumes = []
        signals = []
        discrete_events = []

        event_len = len(events[0])
        flanked_peptide = ("-" * (self.window - 1) + self.peptide +
                           "-" * (self.window - 1))
        num_peaks = len(flanked_peptide) - self.window + 1
        peak_shift = event_len / (num_peaks - 1)

        for event in events:
            event = _normalize(event)
            discretized = []
            #volumes = []
            #signals = []
            #xxx = []

            for i in xrange(0, num_peaks):
                kmer = flanked_peptide[i : i + self.window]
                weights = _aa_to_weights(kmer)
                volume = _avg_volume(weights)

                signal_pos = i * peak_shift
                left = max(0, signal_pos - peak_shift / 2)
                right = min(len(event), signal_pos + peak_shift / 2)
                signal = np.mean(event[left:right])

                volumes.append(volume)
                signals.append(signal)
                discretized.append(signal)
                #xxx.append(signal_pos)
                #print(kmer, weights, signal)
                #print(kmer, signal_pos)

            discrete_events.append(np.array(discretized))

            #plt.plot(xxx, signals)
            #plt.plot(event)
            #plt.show()

        res = linregress(volumes, signals)
        #plt.scatter(volumes, signals)
        #plt.show()
        print(res)
        self.regression = VolSigRegression(res[0], res[1], res[4])
        return discrete_events


    def init_state_distr():
        pass

    def hmm(self, observ_seq):
        num_observ = len(observ_seq)
        num_states = len(init_distr)
        dp_mat = np.zeros((num_states, num_observ))
        backtrack = np.zeros((num_states, num_observ))

        for st in xrange(num_states):
            dp_mat[st][0] = (math.log(init_distr[st]) +
                             math.log(emission_prob(st, observ_seq[0])))
            backtrack[st][0] = None

        #filling dp matrix
        for obs in xrange(1, num_observ):
            for st in xrange(num_states):
                max_val = -float(sys.maxint)
                max_state = None
                for prev_st in xrange(num_states):
                    val = dp_mat[prev_st][obs-1] + math.log(trans_distr[prev_st][st])
                    if val > max_val:
                        max_val = val
                        max_state = prev_st

                dp_mat[st][obs] = (max_val +
                                   math.log(emission_prob(st, observ_seq[obs])))
                backtrack[st][obs] = max_state

        #final state
        final_score = -float(sys.maxint)
        final_state = None
        for st in xrange(num_states):
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
        return states


def _normalize(signal):
    median = np.median(signal)
    return signal / median


AA_SIZE_TRANS = maketrans("GASCTDPNVEQHLIMKRFYW-",
                          "LLLLLLLMMMMMMMHHHHHH-")
def _aa_to_weights(kmer):
    return kmer.translate(AA_SIZE_TRANS)


def _avg_volume(kmer):
    light = kmer.count("L") * 0.1
    medium = kmer.count("M") * 0.16
    heavy = kmer.count("H") * 0.2
    return float(light + medium + heavy) / len(kmer)


def main():
    events = get_data(sys.argv[1])
    nano_hmm = NanoHMM()
    averages = get_averages(events, nano_hmm.average, nano_hmm.flank,
                            nano_hmm.reverse)

    discrete_events = nano_hmm.learn_emissions_distr(averages)
    for obs in discrete_events:
        nano_hmm.hmm(obs)


if __name__ == "__main__":
    main()
