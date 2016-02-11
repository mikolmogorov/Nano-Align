#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Protein identification module
"""

import random

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import distance

import nanoalign.signal_proc as sp

class Identifier(object):
    def __init__(self, blockade_model):
        self.blockade_model = blockade_model
        self.database = None

    def signal_protein_distance(self, signal, peptide):
        theor_signal = self.blockade_model.peptide_signal(peptide)
        return _signals_distance(signal, theor_signal)

    def set_database(self, database):
        """
        Sets protein database. If parameter is None, random
        database is generated
        """
        self.database = database

    def random_database(self, protein, size):
        """
        Generates random database of given size
        with the same length and AA somposition as in the given peptide
        """
        #AAS = "GASCUTDPNVBEQZHLIMKXRFYW"
        weights_list = list(protein)
        database = {}
        database["target"] = protein
        for i in xrange(size):
            random.shuffle(weights_list)
            decoy_name = "decoy_{0}".format(i)
            database[decoy_name] = "".join(weights_list)

        self.database = database

    def identify(self, signal):
        """
        Returns the most similar protein from the database
        """
        return self.rank_database(signal)[0]

    def rank_db_proteins(self, signal):
        """
        Rank database proteins wrt to the similarity to a given signal
        """
        assert self.database is not None

        distances = {}
        discretized = {}

        for prot_id, prot_seq in self.database.items():
            if len(prot_seq) not in discretized:
                discretized[len(prot_seq)] = sp.discretize(signal, len(prot_seq))

            distance = self.signal_protein_distance(discretized[len(prot_seq)],
                                                    prot_seq)
            distances[prot_id] = distance

        return sorted(distances.items(), key=lambda i: i[1])


    """
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
                    errors.append((aa, exp_signal[i] - theor_signal[i]))
        return errors
    """


def _signals_distance(signal_1, signal_2):
    """
    Computes distance between two discrete signals
    """
    return distance.sqeuclidean(signal_1, signal_2)
