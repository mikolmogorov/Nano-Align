#(c) 2015-2016 by Authors
#This file is a part of Nano-Align program.
#Released under the BSD license (see LICENSE file)

"""
Model load/store in pickle format
"""

import pickle

from nanoalign.svr import SvrBlockade
from nanoalign.random_forest import RandomForestBlockade
from nanoalign.mean_volume import MvBlockade


def load_model(filename):
    if filename == "-":
        return MvBlockade()

    dump = pickle.load(open(filename, "rb"))

    if dump.name == SvrBlockade().name:
        model = SvrBlockade()
    elif dump.name == RandomForestBlockade().name:
        model = RandomForestBlockade()

    model.load_from_dump(dump)
    return model


def store_model(model, filename):
    pickle.dump(model.get_dump(), open(filename, "wb"))
