"""General functions that can be used in all modules"""

import pickle

import numpy as np
import scipy as sp


def get_class_counts(ZW):
    """Get number of nodes in each cluster"""
    res = ZW.sum(0)
    if sp.sparse.issparse(ZW):
        res = res.A1
    return res


def encode(ZW, Kzw):
    """Encode cluster membership in a boolean matrix"""
    ND = ZW.shape[0]
    ZW_ = np.zeros((ND, Kzw), dtype='bool')
    ZW_[np.arange(ND), ZW] = True
    return ZW_


def load_model(path):
    """Load a given model from its directory"""
    with open(path + '/model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


# TODO Only usable on dynamic X, check if keep specific in dlbm.py or generalize and keep here.
def check_X(X, is_graph, self_loops, directed=True):
    """
    is_graph: square adjacency matrix?
    directed: whether the graph is directed or not
            this requires a symetric adjacency matrix.
            Only useful for (d)SBM
    """
    T = len(X)

    for t in range(T):
        X_dens_t = to_dense(X[t])
        assert (X_dens_t >= 0).all()
        assert (X_dens_t > 0).any()
        # assert X_dens_t.dtype in [np.int32, np.int64]
        assert X_dens_t.ndim == 2

        if is_graph:
            assert X_dens_t.shape[1] == X_dens_t.shape[0]

        if not self_loops:
            assert (np.diag(X_dens_t) == 0).all()

        if not directed:
            assert np.array_equal(X_dens_t, symmetrize(X_dens_t))


def symmetrize(a):
    """Symmetrize a matrix by keeping only triangular part and copying it on lower triangular"""
    a = np.tril(a)
    return a + a.T - np.diag(a.diagonal())


def get_delta(old, new):
    """
    returns the criterion used to determine
    if the model has converged from the
    old and new likelihoods
    """
    return abs((new - old) / new)


def to_dense(mat):
    """From sp.sparse sparse matrix format to dense Numpy arrays"""
    if sp.sparse.issparse(mat):
        return mat.toarray()

    return mat


def to_sparse(mat):
    """From dense Numpy arrays to sp.sparse sparse matrix format"""
    if not sp.sparse.issparse(mat):
        return sp.sparse.csr_matrix(mat)

    return mat


def compute_mixture_exact_icl(ZW, Kzw, ND):
    """Compute ICL for either lines or columns"""
    from scipy.special import loggamma
    nax = np.newaxis

    cst = (
        loggamma(.5 * Kzw) * (1 - Kzw) -
        .5 * Kzw * (Kzw + 1) * np.log(np.pi) -
        loggamma(.5 * Kzw + ND)
    )

    icl_zw_t1 = loggamma(ZW[0].sum(0) + .5).sum()

    trans = (ZW[:-1, :, :, nax] * ZW[1:, :, nax, :]).sum((0, 1))
    trans_ = trans.sum(1)
    icl_zw_t2 = - loggamma(trans_ + .5 * Kzw).sum() + loggamma(trans + .5).sum()

    return cst + icl_zw_t1 + icl_zw_t2
