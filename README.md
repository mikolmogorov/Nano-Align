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


Data Availability
-----------------

To download the datasets used in the paper, type:

    wget https://github.com/fenderglass/datasets/raw/master/nano-align/nano-align.zip

The package also contains the trained Random Forest and SVR
model for convenience.

Quick Example
-------------

This is a quick example of Nano-Align pipeline, assuming that
you have downloaded the manuscript datasets into 'data' directory.
Please see the detailed usage information below.

Train Random Forest model on H32 nanospectra:

    ./train-model.py rf data/nanospectra/H32.mat h32_rf.pcl

Perform identification of H4 nanospectra:

    ./identify.py data/nanospectra/H4.mat h32_rf.pcl

Plot H32 nanospectra against the RF and MV models:

    ./plotting/models-fit.py data/nanospectra/H32.mat h32_rf.pcl,-

Plot identification p-values as a function of number of nanospectra in a cluster:

    ./plotting/identification-pvalues.py data/nanospectra/H4.mat h32_rf.pcl



Usage
-----

See detailed description of the parameters for each script by
specifying "-h" option.


### train-model.py

Trains the regression model based on Random Forest / SVR,
given nanospectra of a known protein. The output file (model) 
then is then used as an input for other algorithms. 


### identify.py

Performs protein identification and estimates p-values.
It takes trained RF/SVR model as an input.


Visualization scripts
---------------------

There is a number of scripts that can be used to visualize different
features of the data. They are located in the "plotting" directory

### models-fit.py

Plots nanospectra against the corresponding regression models.

### identification-pvalues.py

Plots identification p-values depending on the cluster size

### mixture.py

Plots the frequency distribution of a multiple sets of nanospectra,
originating from different proteins.

### volume-bias.py

Plots volume- or hydrophilicity-related bias of the model.


Scripts for Input Data Manipulation
-----------------------------------

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
