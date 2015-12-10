#!/usr/bin/env python


from __future__ import print_function
import sys

import nanopore.signal_proc as sp


def main():
    if len(sys.argv) != 3:
        print("usage: protein-label.py mat_file prot_sequence")
        return 1

    events = sp.read_mat(sys.argv[1])
    for event in events:
        event.peptide = sys.argv[2]
    sp.write_mat(events, sys.argv[1])
    return 0


if __name__ == "__main__":
    main()
