#!/usr/bin/env python

import sys

import nanopore.signal_proc as sp


def main():
    if len(sys.argv) < 3:
        print("usage: merge-mat.py mat_1[,mat_2..] out_mat")
        return 1

    events = []
    for mat_file in sys.argv[1:-1]:
        events.extend(sp.read_mat(mat_file))

    sp.write_mat(events, sys.argv[-1])


if __name__ == "__main__":
    main()
