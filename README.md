# Dynamic/Constrained Block Models

Version 1.0

## Get started

This package implements algorithms for co-clustering count data based on the Latent Block Model (LBM). There are two main models : a dynamic LBM `dLBM` for data represented as a series of adjacency matrices and a semi-supervised (or constrained) LBM `HLBM` using pairwise constraints in both row and column space. This packages allows sampling data from the models, plotting the data, fitting the models, measuring the clustering performances and analyzing the behavior of the parameters during inference. For more details, see:

### References

Relevant articles & thesis:

- Paul Riverain, Simon Fossier, and Mohamed Nadif. “Poisson Degree Corrected Dynamic Stochastic Block Model.” Advances in Data Analysis and Classification, February 27, 2022. https://doi.org/10.1007/s11634-022-00492-9.
- Paul Riverain, Simon Fossier, and Mohamed Nadif. “Semi-Supervised Latent Block Model with Pairwise Constraints.” Machine Learning 111, no. 5 (May 1, 2022): 1739–64. https://doi.org/10.1007/s10994-022-06137-4.
- Paul Riverain, "Intégration de connaissance métier dans des algorithmes d'apprentissage non supervisé pour le transport ferroviaire", PhD Thesis

## Installing

### Dependencies

**Base:**

    - numpy 1.21
    - scipy
    - numba

**Interactive:**

    - notebook 5.7.10
    - jupyter_contrib_nbextensions
    - jupyter_nbextensions_configurator
    - matplotlib
    - networkx
    - seaborn
    - plotly
    - pandas
    - prince : for Correspondence Analysis
    - nltk  : for notebook of text processing

**Metrics:**

    - sparsebm : for Co-clustering ARI (CARI)

**Initialization:**

    - spherecluster
    - scikit-learn 0.20 : 0.20 because of spherecluster


### Install

*Note*: the tests won't pass if you do not at least install the `[initialization]`, which requires a python3.7 installation.

Within the source folder, run:

    python -m pip install --upgrade pip
    python -m pip install -e .[all]
    jupyter contrib nbextension install --user
    python -m pytest ./dcblockmodels/tests/*short.py  # short tests
    # or
    python -m pytest ./dcblockmodels/tests/*long.py  # long tests

Notes:

- In the previous commands, `python` should be replaced by the Python executable for which you want to install dcblockmodels. As a good practice, you can create a virtual environment.
- The `-e` option in the install command is not required, but it installs the code in editable mode. This allows you to pull code updates on the git server without having to reinstall it.


### Rationale for package dependencies

The package in its core only depends on classical packages (numpy, scipy, numba) and could be compatible with any recent python version. However, the performances of the model depends in part of the initial partition it is given. This initialization can be done with k-means or spherical k-means, the latter being particularly suited to a Poisson LBM with margins. An efficient implementation of k-means is available in `sklearn`, but no implementation of spherical k-means is in `sklearn`. For this reason, we used the implementation of spherical k-means of the package `spherecluster`. This packages uses private methods of a version of `sklearn<=0.20`. This results in a chain of dependencies that implies a requirement to `python 3.7`. Thus, a fully functionning version of the proposed package requires `python 3.7` and `spherecluster`. Note that the `fit()` method of the `HLBM` and `dLBM` have a `given_Z` and `given_W` parameters that allows to give directly row and column initial partitions.


## Testing

Testing such algorithms is complicated. The proposed tests sample data from the model and check that the clustering metrics are not too low. If the tests do not pass because the code breaks, there is a problem. If the tests do not pass beacause the metrics are too low, this could be due to bad luck (complex sampled data or bad initialization) or to a real problem in the code. In the tests, the clustering metrics are compared to thresholds for different setups of sampled data. These thresholds can be changed if needed.


## Documentation

### Code style

The specifics of this codebase in terms of code style are described in [`codestyle.md`](./docs/codestyle.md).

### Code outline

#### Package structure

- the `dcblockmodels` directory contains the models
- the `notebooks` directory contains the sample notebooks
- the `alluvial` directory contains d3.js/HTML/CSS code to create an alluvial diagram for MTR network passenger flow monitoring
- the `saved_models` directory contains the saved fitted models in the form of pickle files


#### Main module

The main module `dcblockmodels` contains:
- the `models` directory that contains the implementations of the models
- the `tests` directory that contains the tests of the models
- `data.py` that contains methods to sample data from static or dynamic LBM/SBM
- `metrics.py` that contains methods measure the quality of a partition of the data or to measure the class separability given the ground truth
- `plot.py` that contains different plotting functions

#### The `utils` submodule

Both `dLBM` and `HLBM` inherit from `blockmodels` that mainly takes care of input check, parameter save for the debug mode and model save. As `dLBM` and `HLBM` share common characteristics, some computations can be done with the same function. The idea is to put as much code as possible in `/dcblockmodels/models/utils` directory in the form of static functions. The `/dcblockmodels/models/utils` directory is separated into 7 files:
- `absent_nodes.py` : creates a class to efficiently deal with the indexes of absent and appearing nodes
- `consensus.py` : performs different kinds of consensus clustering based on a series of partitions of the data
- `e_step.py` : contains the methods for the E-step of the EM algorithm
- `general.py` : contains functions that can be used in all modules
- `m_step.py` : contains the methods for the M-step of the EM algorithm
- `similarity_matrices.py` : contains methods that deal with the pairwise similarity matrices (i.e. the semi-supervision) used for `HLBM`
- `smoothing_schedule.py` : the class that build a smoothing schedule for `dLBM`

### Notations

Conventions used in the code:

    N : number of row nodes
    D : number of column nodes
    ND: general argument, could be N or D

    Kz : number of row clusters
    Kw : number of column clusters
    Kzw: general argument, could be Kz or Kw

    *indexes* :
    i and j are nodes (i.e. row or column),
    t is a timestep,
    k and l are cluster indexes

    Z[t, i, k], W[t, j, l] are respectively the row (resp. columns) variational probabilities in VEM or the cluster indicator matrices in CEM
    ZW : is a general argument and could be Z or W
    qz[t, i, k, k'], qw[t, i, l, l'] : variational transition proba
    qzw : general argument, could be qz or qw

    alpha_beta: general argument, could be alpha or beta
    pi_rho: general argument, could be pi or rho

    ZW_app[t_i, k] : posterior proba of appearing node of size n_appearing_nodes x Kzw
    app_zw_map[t] = {i: t_i}
    the mapping between time step t, node index i and the index t_i used in ZW_app
    app_zw_map is a list of T dicts

## Contributing

If you are interested in contributing to the dcblockmodels project, start by reading the [Contributing guide](/CONTRIBUTING.md).

## License

[MIT License, (c) Thales Group / Université de Paris, 2022](/LICENSE)

* d3.js and d3-sankey.js licensed under BSD 3-clause, Copyright Mike Bostock