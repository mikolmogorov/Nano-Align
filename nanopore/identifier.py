import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as sio
from scipy.spatial import distance
from scipy.stats import spearmanr, norm
from sklearn.svm import SVR
from scipy.interpolate import interp1d



def _signal_score(signal_1, signal_2):
    #return 1 - distance.correlation(signal_1, signal_2)
    #return spearmanr(signal_1, signal_2)[0]
    #return -np.median(np.array(signal_1) - np.array(signal_2))
    return -distance.sqeuclidean(signal_1, signal_2)



    def signal_peptide_score(self, signal, peptide):
        theor_signal = self.peptide_signal(peptide)
        return _signal_score(signal, theor_signal)

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

    def compute_pvalue_raw(self, discr_signal, peptide):
        weights_list = list(peptide)
        misspred = 0
        score = self.signal_peptide_score(discr_signal, peptide)
        decoy_winner = None
        #scores = []
        for x in xrange(10000):
            random.shuffle(weights_list)
            #weights_list = [random.choice(AAS) for _ in xrange(len(peptide))]
            decoy_weights = "".join(weights_list)
            decoy_signal = self.peptide_signal(decoy_weights)
            decoy_score = self.signal_peptide_score(discr_signal, decoy_weights)
            #decoy_score = self.signal_peptide_score(decoy_signal, peptide)
            #scores.append(decoy_score)
            if decoy_score > score:
                decoy_winner = decoy_signal
                misspred += 1

        p_value = float(misspred) / 10000
        #print(np.median(scores), score)
        #print(p_value)
        #self.plot_raw_vs_theory(discr_signal, peptide, decoy_winner)
        return p_value

    def plot_raw_vs_theory(self, discr_signal, peptide, decoy_winner):
        theor_signal = self.peptide_signal(peptide)

        print("Score:", _signal_score(discr_signal, theor_signal))
        if decoy_winner is not None:
            print("Decoy score:", _signal_score(discr_signal, decoy_winner))

        plt.plot(np.repeat(discr_signal, 2), "b-", label="experimental")
        plt.plot(np.repeat(theor_signal, 2), "r-", label="theory")
        if decoy_winner is not None:
            plt.plot(np.repeat(decoy_winner, 2), "g-", label="decoy")
        plt.xlabel("Sampling points")
        plt.ylabel("Normalized signal")
        plt.legend(loc="upper right")
        plt.show()


AAS = "GASCUTDPNVBEQZHLIMKXRFYW"
