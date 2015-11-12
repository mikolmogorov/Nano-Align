#!/usr/bin/env python

from __future__ import print_function
import sys
from nanopore.nanohmm import NanoHMM, aa_to_weights
import nanopore.signal_proc as sp
import random
from collections import defaultdict


def _hamming_dist(str_1, str_2):
    res = 0
    for a, b in zip(str_1, str_2):
        res += int(a != b)
    return res


def _most_common(lst):
    return max(set(lst), key=lst.count)


def indetification_test(events):
    nano_hmm = NanoHMM(PEPTIDE, WINDOW)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)
    nano_hmm.learn_emissions_distr(train_events)

    #build database
    database = []
    peptide_weights = aa_to_weights(nano_hmm.peptide)
    weights_list = list(peptide_weights)
    for _ in xrange(DB_SIZE):
        database.append("".join(weights_list))
        random.shuffle(weights_list)
    hist = defaultdict(int)

    #testing
    misspredicted = 0
    print("Event\tProt_id\tMax_score\tTrue_score\tP-value")
    for num, event in enumerate(test_events):
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
        likelihood, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        true_score = nano_hmm.score(weights, peptide_weights)

        max_score = 0
        chosen_prot = 0
        for prot_id, db_prot in enumerate(database):
            score = nano_hmm.score(db_prot, weights)
            if score > max_score:
                max_score = score
                chosen_prot = prot_id

        if chosen_prot != 0:
            misspredicted += 1
        hist[chosen_prot] += 1
        print("{0}\t{1}\t{2:7.4f}\t{3:7.4f}\t{4:7.2e}"
                .format(num, chosen_prot, max_score, true_score, p_value))

    for prot_id, freq in hist.items():
        print(prot_id, freq)
    print("Misspredicted:", float(misspredicted) / len(test_events))


def relearn(events):
    nano_hmm = NanoHMM(PEPTIDE, WINDOW)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)
    nano_hmm.learn_emissions_distr(train_events)

    scores = {}
    peptide_weights = aa_to_weights(nano_hmm.peptide)
    for num, event in enumerate(test_events):
        print("processing", num, "/", len(test_events))
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
        likelihood, weights = nano_hmm.hmm(event)
        p_value = nano_hmm.compute_pvalue(weights)
        score = nano_hmm.score(weights, peptide_weights)
        print(p_value, score, likelihood)
        scores[num] = score

    len_good = int(len(test_events) * 0.5)
    good_events_ids = sorted(scores, key=scores.get, reverse=True)[:len_good]
    good_events = list(map(lambda i: test_events[i], good_events_ids))
    nano_hmm.learn_emissions_distr(good_events)


def benchmarks(events):
    nano_hmm = NanoHMM(PEPTIDE, WINDOW)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    test_events = sp.get_averages(events, TEST_AVG, FLANK)

    nano_hmm.learn_emissions_distr(train_events)
    correct_weights = aa_to_weights(nano_hmm.peptide)
    print(correct_weights, "\n")
    profile = [[] for x in xrange(len(correct_weights))]

    identified = 0
    identified_raw = 0
    print("Sequence\t\tHMM_score\tFrac_corr\tFit_pvalue\tRaw_pvalue")
    for event in test_events:
        event = sp.normalize(event)
        event = sp.discretize(event, nano_hmm.num_peaks)
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

        nano_hmm.show_fit(event, weights)

    profile = "".join(map(_most_common, profile))
    accuracy = _hamming_dist(profile, correct_weights)
    accuracy = (float(len(correct_weights)) - accuracy) / len(correct_weights)
    print()
    print(profile, accuracy)
    print("Identified:", float(identified) / len(test_events))
    print("Identified raw:", float(identified_raw) / len(test_events))


TRAIN_AVG = 20
TEST_AVG = 5
FLANK = 10
WINDOW = 4
DB_SIZE = 1000

PEPTIDE = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
#PEPTIDE = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
#PEPTIDE = "ARTKQTARKSTGGKAPRKQL"[::-1]


def main():
    events = sp.read_mat(sys.argv[1])
    benchmarks(events)
    #indetification_test(events)
    #relearn(events)


if __name__ == "__main__":
    main()
