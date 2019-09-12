# PersLay: A Simple and Versatile Neural Network Layer for Persistence Diagrams

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
- `tensorflow: 1.13`

Tensorflow can easily be installed using `conda`, `pip`, or following the instructions at https://www.tensorflow.org/install.

Remark: We used `tensorflow-gpu 1.13` in our tests, as it is compatible with `CUDA 10`. If you have access to a GPU and want to use it (that will largely speed up computations), make sure to use a `tensorflow-gpu` version that is compatible with your `CUDA` version. It is likely that other recent versions of `tensorflow` and `tensorflow-gpu` (e.g. `1.12`) should also work.
 
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

In order to generate persistence diagrams, we rely on the GUDHI library, which is a C++/python library whose python version can be installed using 

	$ conda install -c conda-forge gudhi

Otherwise, one can follow the steps at http://gudhi.gforge.inria.fr/python/latest/installation.html.

<!--An additional required package is the `sklearn-tda` package provided at https://github.com/MathieuCarriere/sklearn_tda .
To install this package, run the following instructions in a terminal:

'''
	$ git clone https://github.com/MathieuCarriere/sklearn_tda
	$ cd sklearn_tda
	$ (sudo) pip install .
'''-->

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


The python function `perslay` takes four arguments:

  * the first three ones are: **output** which is the list that will contain the output of PersLay, **name** which is a string defining the name of the PersLay operation for tensorflow, and **diag** which is a numpy array containing the persistence diagrams.

  * the fourth argument is a python dictionary containing the parameters of PersLay. Examples can be found in `tutorialPersLay.ipynb`.
 
In the following description of PersLay parameters, each parameter, or dictionary key, that contains "_init" in its name is optimized and learned by PersLay during training.

  * The following keys are mandatory:

    | **name** | **description** |
    | --- | --- |
    | **layer**              | Either "pm", "im", "ls", "bc", "en", "ex", "rt", "rh". Type of the PersLay layer. "im" is for [persistence images](https://arxiv.org/abs/1507.06217), "ls" is for [persistence landscapes](http://www.jmlr.org/papers/volume16/bubenik15a/bubenik15a.pdf), "ex", "rt" and "rh" are for [structure elements](http://jmlr.org/beta/papers/v20/18-358.html), "pm" is for the original DeepSet layer, defined in [this article](https://arxiv.org/abs/1703.06114), "bc" is for [Betti curves](https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_pdf) and "en" is for [entropy](https://arxiv.org/abs/1803.08304). |
    | **perm_op**            | Either "sum", "mean", "max", "topk". Permutation invariant operation. |
    | **keep**               | Number of top values to keep. Used only if **perm_op** is "topk". |
    | **persistence_weight** | Either "linear", "grid", "gmix" or None. Weight function to be applied on persistence diagram points. If "linear", this function is linear with respect to the distance to the diagonal of the point. If "grid", this function is piecewise-constant and defined with pixel values of a grid. If "gmix", this function is defined as a mixture of Gaussians. If None, no weighting is applied. |

Depending on what **persistence_weight** is, the following additional keys are requested:

  * if **persistence_weight** is "linear":

    | **name** | **description** |
    | --- | --- |
    | **coeff_init**         | Initializer of the coefficient of the linear weight function. It can be either a single value, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(1., 1.). | 
    | **coeff_const**        | Boolean specifying if **coeff_init** is initialized with a value (True) or randomly with tensorflow (False). |
 
  * if **persistence_weight** is "grid":

    | **name** | **description** |
    | --- | --- |
    | **grid_size**          | Grid size of the grid weight function. It is a tuple of integer values, such as (10,10). | 
    | **grid_bnds**          | Grid boundaries of the grid weight function. It is a tuple containing two tuples, each containing the minimum and maximum values of each axis of the plane. Example: ((-0.01, 1.01), (-0.01, 1.01)). | 
    | **grid_init**          | Initializer for the pixel values of the grid weight function. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(1., 1.).| 
    | **grid_const**         | Boolean specifying if **grid_init** is initialized with an array (True) or randomly with tensorflow (False). |

  * if **persistence_weight** is "gmix":

    | **name** | **description** |
    | --- | --- |
    | **gmix_num**           | Number of Gaussian functions of the mixture of Gaussians weight function. |
    | **gmix_m_init**        | Initializer of the means of the mixture of Gaussians weight function. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(1., 1.). |
    | **gmix_m_const**       | Boolean specifying if **gmix_m_init** is initialized with an array (True) or randomly with tensorflow (False). |
    | **gmix_v_init**        | Initializer of the bandwidths of the mixture of Gaussians weight function. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(1., 1.). |
    | **gmix_v_const**       | Boolean specifying if **gmix_v_init** is initialized with an array (True) or randomly with tensorflow (False). Used only if **persistence_weight** is "gmix". |

Depending on what **layer** is, the following additional keys are requested:

  * if **layer** is "pm":

    | **name** | **description** |
    | --- | --- |
    | **peq**                | Sequence of permutation equivariant operations, as defined in [the DeepSet article](). It is a list of tuples of the form (*dim*, *operation*). Each tuple defines a permutation equivariant function of dimension *dim* and second permutation operation *operation* (string, either "max", "min", "sum" or None). Second permutation operation is optional and is not applied if *operation* is set to None. Example: [(150, "max"), (75, None)]. |
    | **weight_init**        | Initializer for the matrices of the permutation equivariant operations. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.).| 
    | **weight_const**       | Boolean specifying if **weight_init** is initialized with a value (True) or randomly with tensorflow (False). | 
    | **bias_init**          | Initializer for the biases of the permutation equivariant operations. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **bias_const**         | Boolean specifying if **bias_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "im":

    | **name** | **description** |
    | --- | --- |
    | **image_size**         | Persistence image size. It is a tuple of integer values, such as (10,10). | 
    | **image_bnds**         | Persistence image boundaries. It is a tuple containing two tuples, each containing the minimum and maximum values of each axis of the plane. Example: ((-0.01, 1.01), (-0.01, 1.01)). |
    | **variance_init**      | Initializer for the bandwidths of the Gaussian functions centered on the persistence image pixels. It can be either a single value, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **variance_const**     | Boolean specifying if **variance_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **cv_layers**          | Sequence of convolution operations to be applied after the permutation invariant operation. Used only if **layer** is "im". It is a list of tuples of the form (*num_filters*, *kernel_size*, *pro*, *dropout*). Each tuple defines a convolution operation, with number of filters *num_filters* (integer), kernel size *kernel_size* (integer), and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(10,3,"bd"), (5,3,"dr",0.8)]. | 
    
  * if **layer** is "ls":

    | **name** | **description** |
    | --- | --- |
    | **num_samples**        | Number of samples of the diagonal that will be evaluated on the persistence landscapes. |
    | **sample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **sample_const**       | Boolean specifying if **sample_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "bc":

    | **name** | **description** |
    | --- | --- |
    | **num_samples**        | Number of samples of the diagonal that will be evaluated on the Betti curves. |
    | **sample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **sample_const**       | Boolean specifying if **sample_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **theta**              | Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points. |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "en":

    | **name** | **description** |
    | --- | --- |
    | **num_samples**        | Number of samples on the diagonal that will be evaluated on the persistence entropies. |
    | **sample_init**        | Initializer of the samples of the diagonal. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **sample_const**       | Boolean specifying if **sample_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **theta**              | Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points. |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "ex":

    | **name** | **description** |
    | --- | --- |
    | **num_elements**       | Number of exponential structure elements that will be evaluated on the persistence diagram points. |
    | **mean_init**          | Initializer of the means of the exponential structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **mean_const**         | Boolean specifying if **mean_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **variance_init**      | Initializer of the bandwidths of the exponential structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **variance_const**     | Boolean specifying if **variance_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "rt":

    | **name** | **description** |
    | --- | --- |
    | **num_elements**       | Number of rational structure elements that will be evaluated on the persistence diagram points. |
    | **mean_init**          | Initializer of the means of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **mean_const**         | Boolean specifying if **mean_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **variance_init**      | Initializer of the bandwidths of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **variance_const**     | Boolean specifying if **variance_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **alpha_init**         | Initializer of the exponents of the rational structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **alpha_const**        | Boolean specifying if **alpha_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **fc_layers**          | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|

  * if **layer** is "rh":

    | **name** | **description** |
    | --- | --- |
    | **num_elements**      | Number of rational hat structure elements that will be evaluated on the persistence diagram points. |
    | **mean_init**         | Initializer of the means of the rational hat structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(0., 1.). |
    | **mean_const**        | Boolean specifying if **mean_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **r_init**            | Initializer of the threshold of the rational hat structure elements. It can be either a numpy array of values, or a random initializer from tensorflow, such as tensorflow.random_uniform_initializer(3., 3.). | 
    | **r_const**           | Boolean specifying if **r_init** is initialized with a value (True) or randomly with tensorflow (False). |
    | **q**                 | Norm parameter. |
    | **fc_layers**         | Sequence of fully-connected operations to be applied after the permutation invariant operation. It is a list of tuples of the form (*dim*, *pro*, *dropout*). Each tuple defines a fully-connected operation, with dimension *dim* (integer) and processing *pro* (string, e.g. "bdr" ---> batch-norm, dropout, relu). If there is a "d" in string, i.e. dropout, the dropout value can be specified with *dro* (float, default 0.9). Example: [(150,"br"), (75, "bd", 0.85)].|


    

    



# Citing PersLay

If you use this code or refer to it, please cite

[1] _PersLay: A Simple and Versatile Neural Network Layer for Persistence Diagrams._
Mathieu Carrière, Frederic Chazal, Yuichi Ike, Théo Lacombe, Martin Royer, Yuhei Umeda.
https://arxiv.org/abs/1904.09378.
