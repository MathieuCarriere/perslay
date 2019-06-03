# PersLay: A Simple and Versatile Neural Network Layer for Persistence Diagrams

This repository provides the implementation of PersLay, a tensorflow 
layer specially desgined to handle persistence diagrams.
This implementation follows the work presented in [1].

It contains a jupyter notebook `tutorialPersLay.ipynb` to try PersLay on graphs and orbits data as in [1].

# Dependencies

All the code was implemented in `Python 3.6`. It is likely that more recent versions would also work.

## Minimal (just using PersLay)

PersLay itself is simply a neural network layer implemented 
using a Tensorflow backend and some `numpy`.
Tests were run on a Linux environment (Ubuntu 18) with the following versions of these libraries 
(it is likely that other versions---especially more recent ones---should also work):
- `numpy: 1.15.4`
- `tensorflow: 1.13`

Tensorflow can easily be installed using `conda`, `pip`, or following the instructions at https://www.tensorflow.org/install .

Remark: We used `tensorflow-gpu 1.13` in our tests, as it is compatible with `CUDA 10`. If you have access to a GPU and want to use it (that will largely speed up computations), make sure to use a `tensorflow-gpu` version that is compatible with your `CUDA` version. It is likely that other recent versions of `tensorflow` and `tensorflow-gpu` (e.g. `1.12`) should also work.
 
## Complete (running the tutorial)

In order to show how PersLay can be used in a learning pipeline on real-life data, we provide a tutorial.
This tutorial has few more dependencies.

### Standard dependencies: 
- `sklearn: 0.20.2`
- `scipy: 1.1.0`
- `pandas: 0.23.4`
- `matplotlib: 3.0.3`
- `h5py: 2.8.0` and `hdf5: 1.10.2` (used to store and load persistence diagrams)

Furthermore, `jupyter notebook` (or equivalent) is needed to run `tutorialPersLay.ipynb`.

### GUDHI and sklearn-tda

In order to produce and process persistence diagrams, we rely on two libraries.

GUDHI is a C++/Python3 library whose Python version can be installed using 

'''
	$ conda install -c conda-forge gudhi
'''

Otherwise, one can follow the steps at http://gudhi.gforge.inria.fr/python/latest/installation.html .

An additional required package is the `sklearn-tda` package provided at https://github.com/MathieuCarriere/sklearn_tda .
To install this package, run the following instructions in a terminal:

'''
	$ git clone https://github.com/MathieuCarriere/sklearn_tda
	$ cd sklearn_tda
	$ (sudo) pip install .
'''

# Organization and content of this repository

The main repository contains two python files that define PersLay.

- `layers.py` defines the different types of layers that can be use when instancing PersLay.
- `archi.py` defines PersLay itself and provide the `model` function that can be used in practice 
(see `tutorial.ipynb` for an example).  

It also contains a Python notebook, `tutorial.ipynb`, that is hopefully easy-of-use and self-contained. 
This notebook relies on the aforementioned libraries and on the various functions defined in the file `./tutorial/utils.py`.

The `/data/` repository contains the files some graph datasets used in [1] experiments. 
Each sub-repository `DATASET` contains a `.conf` file. 
Graphs datasets (aside the `REDDIT` ones, see below) also contains a `/mat/` folder where the different graphs 
(encoded by their adjacency matrix) are stored (`.mat` files). 
Orbit datasets are generated on-the-fly.

### About REDDIT and COLLAB datasets

In [1], we also performed experiments using `COLLAB, REDDIT5K` and `REDDIT12K` datasets.

`REDDIT5K` and `REDDIT12K` datasets are large datasets (5000 and 12000 graphs respectively) 
of large graphs (hundreds of nodes and edges). 
As such, sharing online the adjacency matrices for these datasets is impossible 
(folders are respectively of `18Gb` and `30Gb` size). 
Unfortunately, the url from which we downloaded the initial data, http://www.mit.edu/~pinary/kdd/datasets.tar.gz , appears to be down.

Similarly, `COLLAB` matrices were not included to not overload the repository (about `400Mb`). 

Feel free to contact one of the author if you want more information.

# Citing PersLay

If you use this code or refer to it, please cite
[1] _PersLay: A Simple and Versatile Neural Network Layer for Persistence Diagrams._
Mathieu Carrière, Frederic Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, Yuhei Umeda.