#!/usr/bin/env python

from __future__ import print_function
import sys
import nanopore.signal_proc as sp
from nanopore.nanohmm import NanoHMM, aa_to_weights


def reverse(events, peptide):
    nano_hmm = NanoHMM(peptide)
    train_events = sp.get_averages(events, TRAIN_AVG, FLANK)
    nano_hmm.learn_emissions_distr(train_events)

    peptide_weights = aa_to_weights(nano_hmm.peptide)
    num_reversed = 0
    new_events = []
    print("Samples:", len(events))
    for num, event in enumerate(events):
        norm_trace = sp.normalize(event.trace)
        discr_trace = sp.discretize(norm_trace, nano_hmm.num_peaks)
        likelihood, weights = nano_hmm.hmm(discr_trace)
        p_value = nano_hmm.compute_pvalue(weights)
        score = nano_hmm.score(weights, peptide_weights)
        rev_score = nano_hmm.score(weights, peptide_weights[::-1])
        print(num, p_value, score, rev_score, rev_score > score)

        if rev_score > score:
            #nano_hmm.show_fit(discr_trace, weights)
            num_reversed += 1
            args = list(event.struct)
            args[5] = args[5][::-1]
            new_struct = sp.Struct(*args)
            new_events.append(sp.Event(None, new_struct))
        else:
            new_events.append(event)

    print("Reversed:", num_reversed, "of", len(events))
    return new_events


TRAIN_AVG = 1
FLANK = 50

#PEPTIDE = "SPYSSDTTPCCFAYIARPLPRAHIKEYFYTSGKCSNPAVVFVTRKNRQVCANPEKKWVREYINSLEMS"
#PEPTIDE = "ASVATELRCQCLQTLQGIHPKNIQSVNVKSPGPHCAQTEVIATLKNGRKACLNPASPIVKKIIEKMLNSDKSN"
#PEPTIDE = "ARTKQTARKSTGGKAPRKQL"[::-1]


def main():
    events, peptide = sp.read_mat(sys.argv[1])
    rev_events = reverse(events, peptide)
    sp.write_mat(events, peptide, sys.argv[2])


if __name__ == "__main__":
    main()
