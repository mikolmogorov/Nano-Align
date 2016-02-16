Nano-Align
==========

A set of algorithms for protein identification using a sub-nanopore.


Installation
------------

The package is implemented in Python 2.7 and requires no installation.
However, it depends on some third party Python packages:

* numpy [http://www.numpy.org/]
* scipy [http://www.scipy.org/]
* scikit-learn [http://scikit-learn.org/]
* matplotlib [http://matplotlib.org/]

These packages should be instaled in your system. You can use either 
system package manager (e.g. apt-get in Ubuntu) or *pip* for installation:

    pip install package_name


Data Requirements
-----------------

Nano-Align uses matlab-like files (.mat) with recorded blockade currents
as input. However, there are some modifications comparing to the original
format. Original .mat files could be easily extended to the desired format
using the provided scripts. See the detailed instructions in "Data preparation"
section.


Qick Example
------------

This is a quick example of Nano-Align pipeline. Given
two original ".mat" files with recorded blockades 
(for example, "H32.mat" and "H4.mat"), the commands order
will be as follows:

1. Add protein sequences to .mat files:

    scripts/protein-label.py H32.mat "ARTKQTARK...(H32 sequence)"
    scripts/protein-label.py H4.mat "MSGRGKGGK...(H4 sequence)"

2. Train SVR model:

    train-svr.py H32.mat svr_H32.pcl

3. Normalize blockades directions:

    scripts/flip-blockades.py H4.mat svr_H32.pcl H4_flipped.mat

4. Perform identification:

    identify.py H4_flipped.mat svr_H32.pcl


Usage
-----

The package consist of multple scripts. Three main ones are located
in the root directory:

### train-svr.py

This script trains SVR model using the given blockades signals of a
known protein. The output file (model) then could be used as an
input for other algorithms. Type "train-svr.py -h" for details.


### identify.py

This scipt performs protein identification and estimates p-values.
It requires trained SVR model as an input. Type "identify.py -h"
for the detailed parameter description.


### estimate-length.py

This script measures identifies blockade frequencies, which are
associated with the protein length (and possibly, some other features).


Data Preparation
----------------

Use scripts from "scripts".


Plotting Scripts
----------------

Describe them


Datasets from the Manuscript
----------------------------
