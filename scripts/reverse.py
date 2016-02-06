#!/usr/bin/env python

from __future__ import print_function
import sys
import nanopore.signal_proc as sp
from nanopore.nanohmm import NanoHMM, aa_to_weights


def reverse(events, svr_file):
    clusters = sp.get_averages(events, 1)
    peptide = clusters[0].events[0].peptide
    nano_hmm = NanoHMM(len(peptide), svr_file)
    num_peaks = len(peptide) + 3

    num_reversed = 0
    new_events = []
    print("Samples:", len(events))
    print("Num\tScore\tRev_score\tP-value\tP-value_rev\tNeed_reverse")
    for num, cluster in enumerate(clusters):
        discr_signal = sp.discretize(sp.trim_flank_noise(cluster.consensus),
                                     num_peaks)
        score = nano_hmm.signal_peptide_score(discr_signal, peptide)
        rev_score = nano_hmm.signal_peptide_score(discr_signal, peptide[::-1])
        #p_value = nano_hmm.compute_pvalue_raw(discr_signal, peptide)
        #p_value_rev = nano_hmm.compute_pvalue_raw(discr_signal, peptide[::-1])
        print("{0}\t{1:5.2f}\t{2:5.2f}\t\t{3}"
                .format(num, score, rev_score, rev_score > score))

        new_events.append(cluster.events[0])
        if rev_score > score:
            new_events[-1].eventTrace = new_events[-1].eventTrace[::-1]
            num_reversed += 1

    print("Reversed:", num_reversed, "of", len(events))
    return new_events


def main():
    if len(sys.argv) != 4:
        print("usage: reverse.py mat_in svr_file mat_out")
        return 1

    events = sp.read_mat(sys.argv[1])
    rev_events = reverse(events, sys.argv[2])
    sp.write_mat(rev_events, sys.argv[3])

    return 0


if __name__ == "__main__":
    main()
