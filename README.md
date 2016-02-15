Nano-Align
==========

A set of algorithms for protein identification using a sub-nanopore.


Installation
------------

The package is implemented in Python 2.7 and requires no installation.
However, it depends on some third party packages:

* Numpy [http://www.numpy.org/]
* Scipy [http://www.scipy.org/]
* Scikit-Learn [http://scikit-learn.org/]
* Matplotlib [http://matplotlib.org/]

These packages should be instaled in your system. You can either use
system package manager (e.g. apt-get in Ubuntu) or *pip* for installation:

    pip install package_name


Data Requirements
-----------------

Nano-Align uses matlab-like files (*.mat) with recorded blockade currents
as input. However, there are some modifications comparing to the original
format. Original .mat files could be easily extended to the desired format
using the provided scripts. See the detailed instructions in "Data preparation"
section.


Usage
-----

The package consist of multple scripts. Three main ones are located
in the root directory:

### train-svr.py

This script trains SVR model using the given blockades signals from a
known protein. The output file (model) then could be used as an
input for other algorithms. Type "train-svr.py -h" for details.


### identify.py

This scipt performs protein identification and estimates p-values.
It requires trained SVR model as an input. Type "identify.py -h"
for the detailed parameter description.


### estimate-length.py

This script measures identifies blockade frequencies, which are
associated with the protein length (and possibly, some other features).


Data preparation
----------------

Use scripts from "scripts".
