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


Quick Example
-------------

This is a quick example of Nano-Align pipeline. Given
two original ".mat" files with recorded blockades 
(for example, "H32.mat" and "H4.mat"), the commands order
will be as follows:

Add protein sequences to .mat files:

    scripts/protein-label.py H32.mat "ARTKQTARK...(H32 sequence)"
    scripts/protein-label.py H4.mat "MSGRGKGGK...(H4 sequence)"

Train SVR model:

    train-svr.py H32.mat svr_H32.pcl

Normalize blockades directions:

    scripts/flip-blockades.py H4.mat svr_H32.pcl H4_flipped.mat

Perform identification:

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

This script measures blockade frequencies, which are
associated with the protein length (and possibly, some other features).


Data Preparation
----------------

The scripts located in "scripts" directory provide some extra
functionality for the data preprocessing and analysis.

### cut-protein-db.py

Creates a protein database with the certain protein lengths from
a bigger FASTA database (such as human proteome).

### flip-blockades.py

Given blockades singals and trained SVR model, for each blockade
determines if was recorded from straight (N- to C-terminus) or
reverse protein translocation. Reverse translocations are then flipped.

### merge-mats.py

Merges multiple .mat files into one.

### protein-label.py

Adds protein sequence labels to .mat file - a prerequisite for
the further analysis.


Plotting Scripts
----------------

Scripts from "plotting" directory could be used for drawing plots
(similar to ones from the manuscript).

### blockades-freq.py

Plots blocakdes frequencies for different dataset to highlight the difference.

### identification-pvalues.py

Plots p-values of protein identification for the different cluster sizes.

### models-fit.py

Plots emperical blockades versus theoretical models for the visual comparison.

### volume-bias.py

Plots volume-related or hydrophillicity-related bias based on the
difference between empirical and theoretical traces.


Datasets from the Manuscript
----------------------------

Here we describe the exact datasets used in the manuscript, which
could be used for the reproduction of the results. The exact commands that
were used to get the reults are given in "Quick Example" section.
The plotting scripts are located in "plotting" directory.

* "H3.2" -- *ZD350_H32_D5.mat*
* "H4" -- a union of *ZD349_H4_D3.mat*, *ZD349_H4_D4.mat* and *ZD349_H4_D5.mat*
* "CCL5" -- *ZD158_CCL5.mat*
* "H3" -- *ZD243_H3N.mat* 
* "H3.3" -- *ZD350_H33_D2.mat*
