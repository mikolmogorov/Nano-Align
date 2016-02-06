#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
This module defines blockade structure and IO functions
"""

import scipy.io as sio
import numpy as np


class Blockade(object):
    def __init__(self, fileTag, StartPoint, ms_Dwell, pA_Blockade, openPore,
                 eventTrace, correlation, peptide):
        self.fileTag = fileTag
        self.StartPoint = StartPoint
        self.ms_Dwell = ms_Dwell
        self.pA_Blockade = pA_Blockade
        self.openPore = openPore
        self.eventTrace = eventTrace
        self.correlation = correlation
        self.peptide = peptide


class BlockadeCluster(object):
    def __init__(self, consensus, blockades):
        self.consensus = consensus
        self.blockades = blockades


def read_mat(filename):
    """
    Load blockades from mat file
    """
    mat_file = sio.loadmat(filename)
    struct = mat_file["Struct"][0][0]
    event_traces = struct["eventTrace"]
    num_samples = event_traces.shape[1]

    blockades = []
    for sample_id in xrange(num_samples):
        file_tag = struct["fileTag"][sample_id]
        start_point = float(struct["StartPoint"].squeeze()[sample_id])
        dwell = float(struct["ms_Dwell"].squeeze()[sample_id])
        pa_blockade = float(struct["pA_Blockade"].squeeze()[sample_id])
        open_pore = float(struct["openPore"].squeeze()[sample_id])
        correlation = float(struct["correlation"].squeeze()[sample_id])
        try:
            peptide = str(struct["peptide"][sample_id]).strip()
        except IndexError:
            peptide = None

        trace = np.array(event_traces[:, sample_id])

        out_struct = Blockade(file_tag, start_point, dwell, pa_blockade,
                              open_pore, trace, correlation, peptide)
        blockades.append(out_struct)

    return blockades


def write_mat(blockades, filename):
    """
    Store blockades in matlab format
    """
    dtype = [("fileTag", "O"), ("StartPoint", "O"), ("ms_Dwell", "O"),
             ("pA_Blockade", "O"), ("eventTrace", "O"), ("openPore", "O"),
             ("correlation", "O"), ("peptide", "O")]
    file_tag_arr = np.array(map(lambda e: e.fileTag, blockades))
    peptide_arr = np.array(map(lambda e: e.peptide, blockades))
    start_arr = np.array(map(lambda e: e.StartPoint, blockades))
    dwell_arr = np.array(map(lambda e: e.ms_Dwell, blockades))
    pa_blockade_arr = np.array(map(lambda e: e.pA_Blockade, blockades))
    open_pore_arr = np.array(map(lambda e: e.openPore, blockades))
    event_trace_arr = np.array(map(lambda e: e.eventTrace, blockades))
    corr_arr = np.array(map(lambda e: e.correlation, blockades))

    struct = (file_tag_arr, [start_arr], [dwell_arr], [pa_blockade_arr],
              np.transpose(event_trace_arr), [open_pore_arr],
              [corr_arr], peptide_arr)
    sio.savemat(filename, {"Struct" : np.array([[struct]], dtype=dtype)})
