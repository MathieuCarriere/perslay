# PersLay: a neural network layer for persistence diagrams and new graph topological signatures

__Authors:__  Mathieu Carriere, Theo Lacombe, Martin Royer

__Note:__ This is an alpha version of PersLay. 
Feel free to contact the authors for any remark, suggestion, bug, etc.

This repository provides the implementation of PersLay, a tensorflow layer specifically desgined to handle persistence diagrams.
This implementation follows the work presented in [1].

It contains a jupyter notebook `tutorialPersLay.ipynb` to try PersLay on graphs and orbits data as in [1].

PersLay can be installed by running the following instructions in a terminal:

	$ git clone https://github.com/MathieuCarriere/perslay
	$ cd perslay
	$ (sudo) pip install .

# Dependencies

All the code was implemented in `python 3.6`. It is likely that more recent versions would also work.

## Minimal (just using PersLay)

PersLay itself is simply a neural network layer implemented using a tensorflow backend and some `numpy`.
Tests were run on a Linux environment (Ubuntu 18.04) with the following versions of these libraries 
(it is likely that other versions---especially more recent ones---should also work):
- `numpy: 1.15.4`
- `tensorflow: 2.0`

Tensorflow can easily be installed using `conda`, `pip`, or following the instructions at https://www.tensorflow.org/install.

Remark: We used `tensorflow-gpu 1.13` in an older version of the code, please check old releases of PersLay on the GitHub repo if you want to keep using Tensorflow versions less than 2.0.
<!-- our tests, as it is compatible with `CUDA 10`. If you have access to a GPU and want to use it (that will largely speed up computations), make sure to use a `tensorflow-gpu` version that is compatible with your `CUDA` version. It is likely that other recent versions of `tensorflow` and `tensorflow-gpu` (e.g. `1.12`) should also work. -->
 
## Complete (running the tutorial)

In order to show how PersLay can be used in a learning pipeline on real-life data, we provide a jupyter notebook.
This notebook has a few more dependencies.

### Standard dependencies: 
- `sklearn: 0.20.2`
- `scipy: 1.1.0`
- `pandas: 0.23.4`
- `matplotlib: 3.0.3`
- `h5py: 2.8.0` and `hdf5: 1.10.2` (used to store and load persistence diagrams)

Furthermore, `jupyter notebook` (or equivalent) is needed to run `tutorialPersLay.ipynb`.

### GUDHI

In order to generate and/or preprocess persistence diagrams, we rely on the GUDHI library, which is a C++/python library whose python version can be installed using 

	$ conda install -c conda-forge gudhi

Otherwise, one can follow the steps at http://gudhi.gforge.inria.fr/python/latest/installation.html.

# Organization and content of this repository

The main repository contains the python file `perslay.py` that defines the PersLay operation and the different types of layers that can be used with it.

Moreover, the folder `tutorial` contains the python notebook `tutorialPersLay.ipynb`, that reproduces the experiments in [1], contains an example of neural network using PersLay that can be used as a template for other experiments, and is hopefully easy to use and self-contained. 
This notebook relies on the aforementioned libraries and on the various functions defined in the files `utils.py`, `preprocessing.py` and `expe.py`.
The `/data/` repository contains the graph datasets used in the experiments of [1].
<!--Each sub-repository `DATASET` contains a `.conf` file.--> 
Graphs datasets (aside the `COLLAB` and `REDDIT` ones, see below) also contains a `/mat/` folder where the different graphs 
(encoded by their adjacency matrix) are stored (`.mat` files). Orbit datasets are generated on-the-fly.

Finally, you will also find the python notebook `visuPersLay.ipynb` in the `tutorial` folder, which contains examples of PersLay computations on a single persistence diagram. This notebook shows how the usual persistence vectorizations, such as persistence landscapes or images, can be retrieved as special cases of the PersLay architecture. 


### About REDDIT and COLLAB datasets

In [1], we also performed experiments using `COLLAB, REDDIT5K` and `REDDIT12K` datasets.

`REDDIT5K` and `REDDIT12K` datasets are large datasets (5,000 and 12,000 graphs respectively) 
of large graphs (hundreds of nodes and edges). 
As such, sharing online the adjacency matrices for these datasets is impossible 
(folders are respectively of `18Gb` and `30Gb` size). 
Similarly, `COLLAB` matrices were not included to not overload the repository (about `400Mb`).
Unfortunately, the URL from which we downloaded the initial data, http://www.mit.edu/~pinary/kdd/datasets.tar.gz, appears to be down.

Feel free to contact one of the authors if you want more information.

# How to call and use PersLay


The `perslay` package contains a class `PerslayModel` which implements a Tensorflow / Keras model, and which is initialized with four arguments:

  * **name** which is the Tensorflow name of the model,

  * **diagdim** which is the dimension of the persistence diagram points (usually 2),

  * **rho** which is a Tensorflow / Keras model used for postprocessing the concatenated vectorized persistence diagrams of all channels (see the architecture described in [1]). Use "identity" if you don't want to postprocess.

  * **perslay_parameters**, which is a python dictionary containing the parameters of PersLay. Examples can be found in `tutorialPersLay.ipynb`.
 
In the following description of PersLay parameters, each parameter, or dictionary key, that contains `_init` in its name is optimized and learned by PersLay during training. If you do not want to optimize the vectorization, set the keys **train_vect** and **train_weight** to False.

  * The following keys are mandatory:

    | **name** | **description** |
    | --- | --- |
    | **layer**              | Either "PermutationEquivariant", "Image", "Landscape", "BettiCurve", "Entropy", "Exponential", "Rational" or "RationalHat". Type of the PersLay layer. "Image" is for [persistence images](https://arxiv.org/abs/1507.06217), "Landscape" is for [persistence landscapes](http://www.jmlr.org/papers/volume16/bubenik15a/bubenik15a.pdf), "Exponential", "Rational" and "RationalHat" are for [structure elements](http://jmlr.org/beta/papers/v20/18-358.html), "PermutationEquivariant" is for the original DeepSet layer, defined in [this article](https://arxiv.org/abs/1703.06114), "BettiCurve" is for [Betti curves](https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_pdf) and "Entropy" is for [entropy](https://arxiv.org/abs/1803.08304). |
    | **perm_op**            | Either "sum", "mean", "max", "topk". Permutation invariant operation. |
    | **keep**               | Number of top values to keep. Used only if **perm_op** is "topk". |
    | **pweight**            | Either "power", "grid", "gmix" or None. Weight function to be applied on persistence diagram points. If "power", this function is a (trainable) coefficient times the distances to the diagonal of the points to a certain power. If "grid", this function is piecewise-constant and defined with pixel values of a grid. If "gmix", this function is defined as a mixture of Gaussians. If None, no weighting is applied. |
    | **final_model**        | A Tensorflow / Keras model used to postprocess the persistence diagrams in each channel. Use "identity" if you don't want to postprocess. |

Depending on what **pweight** is, the following additional keys are requested:

  * if **pweight** is "power":

    | **name** | **description** |
    | --- | --- |
    | **pweight_init**         | Initializer of the coefficient of the power weight function. It can be either a single value, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). | 
    | **pweight_power**        | Integer used for exponentiating the distances to the diagonal of the persistence diagram points. |
 
  * if **pweight** is "grid":

    | **name** | **description** |
    | --- | --- |
    | **pweight_size**          | Grid size of the grid weight function. It is a tuple of integer values, such as (10,10). | 
    | **pweight_bnds**          | Grid boundaries of the grid weight function. It is a tuple containing two tuples, each containing the minimum and maximum values of each axis of the plane. Example: ((-0.01, 1.01), (-0.01, 1.01)). | 
    | **pweight_init**          | Initializer for the pixel values of the grid weight function. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.).|

  * if **pweight** is "gmix":

    | **name** | **description** |
    | --- | --- |
    | **pweight_num**           | Number of Gaussian functions of the mixture of Gaussians weight function. |
    | **pweight_init**          | Initializer of the means and variances of the mixture of Gaussians weight function. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |

Depending on what **layer** is, the following additional keys are requested:

  * if **layer** is "PermutationEquivariant":

    | **name** | **description** |
    | --- | --- |
    | **lpeq**                | Sequence of permutation equivariant operations, as defined in [the DeepSet article](). It is a list of tuples of the form (*dim*, *operation*). Each tuple defines a permutation equivariant function of dimension *dim* and second permutation operation *operation* (string, either "max", "min", "sum" or None). Second permutation operation is optional and is not applied if *operation* is set to None. Example: [(150, "max"), (75, None)]. |
    | **lweight_init**        | Initializer for the weight matrices of the permutation equivariant operations. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.).| 
    | **lbias_init**          | Initializer for the biases of the permutation equivariant operations. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **lgamma_init**         | Initializer for the Gamma matrices of the permutation equivariant operations. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.).|

  * if **layer** is "Image":

    | **name** | **description** |
    | --- | --- |
    | **image_size**         | Persistence image size. It is a tuple of integer values, such as (10,10). | 
    | **image_bnds**         | Persistence image boundaries. It is a tuple containing two tuples, each containing the minimum and maximum values of each axis of the plane. Example: ((-0.01, 1.01), (-0.01, 1.01)). |
    | **lvariance_init**      | Initializer for the bandwidths of the Gaussian functions centered on the persistence image pixels. It can be either a single value, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 3.). | 
    
  * if **layer** is "Landscape":

    | **name** | **description** |
    | --- | --- |
    | **lsample_num**        | Number of samples of the diagonal that will be evaluated on the persistence landscapes. |
    | **lsample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |

  * if **layer** is "BettiCurve":

    | **name** | **description** |
    | --- | --- |
    | **lsample_num**        | Number of samples of the diagonal that will be evaluated on the Betti curves. |
    | **lsample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **theta**              | Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points. |
    
  * if **layer** is "Entropy":

    | **name** | **description** |
    | --- | --- |
    | **lsample_num**        | Number of samples on the diagonal that will be evaluated on the persistence entropies. |
    | **lsample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **theta**              | Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points. |
    
  * if **layer** is "Exponential":

    | **name** | **description** |
    | --- | --- |
    | **lnum**       | Number of exponential structure elements that will be evaluated on the persistence diagram points. |
    | **lmean_init**          | Initializer of the means of the exponential structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **lvariance_init**      | Initializer of the bandwidths of the exponential structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    
  * if **layer** is "Rational":

    | **name** | **description** |
    | --- | --- |
    | **lnum**       | Number of rational structure elements that will be evaluated on the persistence diagram points. |
    | **lmean_init**          | Initializer of the means of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **lvariance_init**      | Initializer of the bandwidths of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **lalpha_init**         | Initializer of the exponents of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    
  * if **layer** is "RationalHat":

    | **name** | **description** |
    | --- | --- |
    | **lnum**      | Number of rational hat structure elements that will be evaluated on the persistence diagram points. |
    | **lmean_init**         | Initializer of the means of the rational hat structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **lr_init**            | Initializer of the threshold of the rational hat structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **q**                 | Norm parameter. |

    



# Citing PersLay

If you use this code or refer to it, please cite

[1] _PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures._
Mathieu Carriere, Frederic Chazal, Yuichi Ike, Theo Lacombe, Martin Royer, Yuhei Umeda
Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, PMLR 108:2786-2796, 2020.
