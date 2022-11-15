"""
Methods that deal with the pairwise similarity matrices (i.e. the semi-supervision) used for HLBM
"""

from itertools import combinations, product

import numpy as np
import scipy as sp
from numba import jit, prange


def check_similarity_matrix(S):
    """Checks on similarity matrix correctness"""
    def is_sparse_mat_symetric(S):
        S_res = S - S.T
        return np.all(np.abs(S_res.data) < 1e-3)

    assert is_sparse_mat_symetric(S)
    assert (S[np.diag_indices_from(S)] == 0.).all()


def init_transform(X, S, p):
    """
    Returns the np.array X transformed according
    to a multiplicative initialization technique
    that uses the similarity matrix S. The order p
    can be used to propagate the similarity relationship
    to p-neighborhoods.
    Note that X_transformed is a dense array, such that
    running kmeans or skmeans on it can be inefficient
    in a high-dimensional setting
    """
    # transform similarity matrix S into a stochastic matrix W
    W = np.clip(S.copy().toarray(), 0., None)
    np.fill_diagonal(W, 1.)

    with np.errstate(divide='ignore'):
        W_ = W.sum(1)
        np.divide(W, W_, where=(W_ > 0.), out=W)
    np.nan_to_num(W, copy=False, nan=0., posinf=0., neginf=None)

    if p > 1:
        with np.errstate(under='ignore'):
            W = np.linalg.matrix_power(W, p)

    X_transformed = W @ X

    np.clip(X_transformed, 1e-10, None, out=X_transformed)
    return X_transformed


def build_S_mask(N, frac):
    """
    Returns a random N x N symmetric boolean matrix
    without self loops such that int(frac * N * (N - 1))
    of its elements are equal to True
    Used for sampling a given fraction of all the
    possible constraints
    """
    def i_max(k):
        return (k + 1) * N - int((k + 1) * (k + 2) / 2) - 1
    i_max_ = [i_max(k) for k in range(N)]

    S = np.zeros((N, N), dtype='bool')
    n_pairs = int(N * (N - 1) / 2)
    n_pairs_sampled = int(frac * n_pairs)
    indexes = np.random.choice(n_pairs, size=n_pairs_sampled, replace=False)
    indexes = sorted(indexes)

    # k, l the row and columns indexes corresponding
    # to the edge number 'ind' in S
    # k_0 the current row index, to speed up
    # computations, since 'indexes' is sorted
    k_0 = 0
    for ind in indexes:
        for k in range(k_0, N):
            i_min_k = i_max_[k - 1] + 1 if k != 0 else 0
            i_max_k = i_max_[k]
            if i_min_k <= ind <= i_max_k:
                l = ind - i_min_k + (k + 1)
                k_0 = k
                # print(ind, k, l, i_max_k, i_min_k)
                S[k, l], S[l, k] = True, True
                break
    return S


def build_S_strat(ZW, frac, frac_noise=0., path_only=False):
    """
    returns a similarity matrix build from the
    true classes ZW, by sampling a fraction frac
    of the nodes of each class and adding noise
    to a fraction frac_noise of the nodes
    """
    ND = ZW.shape[0]
    Kzw = np.unique(ZW).shape[0]
    S = np.zeros((ND, ND))
    for kzw in range(Kzw):
        ind_kzw = np.where(ZW == kzw)[0]
        nb_nodes = int(frac * ind_kzw.shape[0])
        ind_sampled = np.random.choice(ind_kzw, nb_nodes, replace=False)
        if not path_only:
            S[np.ix_(ind_sampled, ind_sampled)] = 1
        else:
            for i_1, i_2 in zip(ind_sampled[:-1], ind_sampled[1:]):
                S[i_1, i_2] = 1
                S[i_2, i_1] = 1
    if frac_noise > 0.:
        nb_nodes_random = int(frac_noise * ND)
        ind_rnd = np.random.choice(ND, nb_nodes_random, replace=False)
        S[np.ix_(ind_rnd, ind_rnd)] = 1 - S[np.ix_(ind_rnd, ind_rnd)]

    S[np.diag_indices(ND)] = 0.
    assert ((S - S.T) == 0).all()
    return S


def build_S(ZW, frac, frac_noise=None):
    """
    returns a similarity matrix build from the
    true classes ZW, by sampling a fraction frac
    of the nodes of each class

    consider using build_S_sparse
    """
    ND = ZW.shape[0]
    clusters = np.unique(ZW)

    # S_ZW[i, j] = 1 if i and j are in the same
    # cluster in the partition ZW otherwise S_ZW[i, j] = -1
    S_ZW = np.zeros((ND, ND), dtype='int')
    for k in clusters:
        ind_k = np.where(ZW == k)[0]
        z_k = np.zeros((ND, 1), dtype='int')
        z_k[ind_k, :] = True
        S_ZW += z_k.dot(z_k.T)

    S_ZW[S_ZW == 0] = -1
    S_mask = build_S_mask(ND, frac)
    S = S_ZW * S_mask

    if frac_noise is not None:
        S_mask_noise = build_S_mask(ND, frac_noise)
        S = (S * (~S_mask_noise) - S_mask_noise * S)

    S[np.diag_indices(ND)] = 0.
    assert ((S - S.T) == 0).all()
    return S


def build_S_ssl(y):
    """
    Given an array y of size N with integer values, where
    y[i] = k : means that observation i is in cluster k
    and y[i] = -1 : means that we have no information about
    the cluster of observation i;
    returns a N x N array S where:
        S[i, j] = 1 if i and j are in the same cluster,
        S[i, j] = -1 if they are in different clusters
        S[i, j] = 0 if we have no information
    Allows to compare classical semi-supervised
    approaches to pairwise semi-supervised clustering.
    """
    N = y.shape[0]
    clusters = np.unique(y)

    S = np.zeros((N, N), dtype='int')
    for k in clusters:
        if k >= 0:
            ind_k = np.where(y == k)[0]
            y_k = np.zeros((N, 1), dtype='int')
            y_k[ind_k, :] = True
            S += y_k.dot(y_k.T)

    S[S == 0] = -1
    ind_masked = np.where(y == -1)[0]
    S[ind_masked] = 0
    S[:, ind_masked] = 0
    S[np.diag_indices(N)] = 0.
    assert ((S - S.T) == 0).all()
    return S


@jit(nopython=True)
def i_max_sparse(N, k):
    """Temporary value computation for i_max_ (for numba parallel computation)"""
    return (k + 1) * N - int((k + 1) * (k + 2) / 2) - 1


@jit(nopython=True)
def indexes_S_mask_sparse(N, frac):
    """
    returns a random N x N symmetric boolean matrix
    without self loops such that int(frac * N * (N - 1))
    of its elements are equal to True
    """
    i_max_ = [i_max_sparse(N, k) for k in prange(N)]  # pylint: disable=not-an-iterable

    n_pairs = int(N * (N - 1) / 2)
    n_pairs_sampled = int(frac * n_pairs)
    indexes = np.random.choice(n_pairs, size=n_pairs_sampled, replace=False)
    indexes = sorted(indexes)

    row_indexes, col_indexes = [], []

    # k, l the row and columns indexes corresponding
    # to the edge number 'ind' in S
    # k_0 the current row index, to speed up
    # computations, since 'indexes' is sorted
    k_0 = 0
    for ind in indexes:
        for k in range(k_0, N):
            i_min_k = i_max_[k - 1] + 1 if k != 0 else 0
            i_max_k = i_max_[k]
            if i_min_k <= ind <= i_max_k:
                l = ind - i_min_k + (k + 1)
                k_0 = k
                # print(ind, k, l, i_max_k, i_min_k)
                # S[k, l], S[l, k] = True, True
                row_indexes.append(k)
                col_indexes.append(l)

                row_indexes.append(l)
                col_indexes.append(k)
                break

    return row_indexes, col_indexes


def build_S_mask_sparse(n, row_indexes, col_indexes):
    """Build a mask on a sparse matrix"""
    S = sp.sparse.dok_matrix((n, n), dtype='bool')
    dic_indexes = zip(row_indexes, col_indexes)
    dic_values = (1 for _ in range(len(row_indexes)))
    update_dic = dict(zip(dic_indexes, dic_values))
    S._update(update_dic)  # pylint: disable=W0212
    return S.tocsr()


def build_S_sparse(ZW, frac, stratified=False):
    """
    returns a similarity matrix build from the
    true classes ZW, by sampling a fraction frac
    of all the possible ML and CL relationships (stratified = True)
    or by sampling a fraction frac of the nodes of each class

    Caution : the scales for frac are very different depending
    on the value of wether stratified is true or not
    """
    ND = ZW.shape[0]
    clusters = np.unique(ZW)

    # S_ZW[i, j] = 1 if i and j are in the same
    # cluster in the partition ZW otherwise S_ZW[i, j] = -1
    S_ML = sp.sparse.csr_matrix((ND, ND), dtype='bool')
    S_CL = sp.sparse.csr_matrix((ND, ND), dtype='bool')

    # build S_ML and S_CL that conatain all relationships
    if stratified:
        # stores sampled indices for S
        inds = {}
        for k in clusters:
            ind_k = np.where(ZW == k)[0]
            n_nodes_k = len(ind_k)
            n_nodes_sampled = int(frac * n_nodes_k)
            ind_k = np.random.choice(ind_k, size=n_nodes_sampled, replace=False)
            inds[k] = ind_k

        # build S_ML and S_CL
        for k, ind_k in inds.items():
            z_k = np.zeros((ND, 1), dtype='bool')
            z_k[ind_k, :] = True

            # z_k_bar[i] = 1 if i is known not to be in cluster k, 0 otherwise
            z_k_bar = np.zeros((ND, 1), dtype='bool')
            for k_, ind_k_ in inds.items():
                if k_ != k:
                    z_k_bar[ind_k_, :] = True

            z_k = sp.sparse.csr_matrix(z_k)
            z_k_bar = sp.sparse.csr_matrix(z_k_bar)
            S_ML += z_k * z_k.T
            S_CL += z_k * z_k_bar.T

        S = (S_ML.astype('int') - S_CL.astype('int'))
        S[np.diag_indices_from(S)] = 0
    else:
        for k in clusters:
            ind_k = np.where(ZW == k)[0]
            z_k_ = np.zeros((ND, 1), dtype='bool')
            z_k_[ind_k, :] = True

            # z_k[i] = 1 if i is known to be in cluster k, 0 otherwise
            # z_k_bar[i] = 1 if i is known not to be in cluster k, 0 otherwise
            z_k = sp.sparse.csr_matrix(z_k_)
            z_k_bar = sp.sparse.csr_matrix((1 - z_k_).astype('bool'))
            S_ML += z_k * z_k.T
            S_CL += z_k * z_k_bar.T

        # sample a fraction frac of the relatioships
        # and builds the integer valued matrix S
        # S_mask determines which relatiships are sampled
        row_indexes, col_indexes = indexes_S_mask_sparse(ND, frac)
        S_mask = build_S_mask_sparse(ND, row_indexes, col_indexes)
        S = sp.sparse.csr_matrix.multiply(
            S_ML.astype('int') - S_CL.astype('int'),
            S_mask
        )
    return S


def must_link_and_cannot_link_closure(S):
    """
    Given a must link and cannot link matrix S
    such that S[i, j] = 1 for ML, S[i, j] = -1 for CL
    and S[i, j] = 0 otherwise, returns a matrix with
    the transitive and reflexive closure of the ML
    relationship and the closure of the CL relationship.
    """
    def transitive_closure(list_tuples):
        closure = set(list_tuples)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return closure

    if sp.sparse.issparse(S):
        S = S.toarray()

    assert S.shape[0] == S.shape[1]
    assert (S.T == S).all()
    assert np.isin(np.unique(S), np.array([-1, 0, 1])).all()

    # builds S_res that contains the transitive closure
    # of the ML relationship
    S_tu = np.triu(S)
    t_clos_ml = transitive_closure(list(zip(*np.where(S_tu > 0.))))
    S_res = np.zeros_like(S)
    for i, j in t_clos_ml:
        S_res[i, j], S_res[j, i] = 1., 1.

    # builds the neighborhoods given by the transitive
    # closure of the ML relationship
    N = S.shape[0]
    S_ = S_res + np.eye(N)
    neigh_ml = []
    for i in range(N):
        neigh = np.where(S_[i] > 0.)[0]
        new_neigh = np.array([len(np.intersect1d(n, neigh)) == 0 for n in neigh_ml]).all()
        if new_neigh:
            neigh_ml.append(neigh)

    # add the CL relatioships between all pairs
    # of nodes of two neighborhoods if there is
    # a CL relationship between two nodes of
    # these neighborhoods
    for n1, n2 in combinations(neigh_ml, 2):
        if (S[np.ix_(n1, n2)] < 0.).any():
            for i, j in product(n1, n2):
                S_res[i, j], S_res[j, i] = -1, -1

    return S_res


def normalize_S(S):
    """
    Normalizes the similarity matrix S
    as presented in the paper
    """
    S_ = S.sum(0)
    S_inv = np.zeros((S.shape[0]))
    np.divide(1, S_, where=(S_ > 0.), out=S_inv)
    root_D_inv = np.diag(np.sqrt(S_inv))
    return root_D_inv @ S @ root_D_inv


def similarity_discordance(ZW, S, weighted):
    """
    Counts the proportion of ML or CL constraints
    that are not satisfied in the partition ZW.
    If ZW is the true partition, the returned value
    represents the discordance between the true classes
    and the given similarity information.
    If ZW is the partition returned by the algorithm
    without similarity matrix, the returned value
    represents the information brought by the given
    similarity matrix.
    If ZW is the partition returned by the algorithm
    with similarity matrix, the returned value
    represents the proportion of constraints from the
    similarity matrix that are not respected after
    regularisation.
    """
    clusters = np.unique(ZW)
    ND = ZW.shape[0]
    not_S_ZW = sp.sparse.csr.csr_matrix((ND, ND), dtype='bool')
    # not_S_ZW[i, j] = True if ZW[i] != ZW[j] else False
    for k in clusters:
        ind_k = np.where(ZW == k)[0]
        z_k0 = np.zeros((ND, 1), dtype='bool')
        z_k0[ind_k, :] = True
        z_k = sp.sparse.csr.csr_matrix(z_k0)
        not_z_k = sp.sparse.csr.csr_matrix(~z_k0)
        not_S_ZW += z_k.dot(not_z_k.T)

    S_ZW = sp.sparse.csr.csr_matrix(~not_S_ZW.toarray())  # pylint: disable=E1130

    S_ml = S.copy()
    S_ml[S_ml < 0.] = 0.
    S_cl = S.copy()
    S_cl[S_cl > 0.] = 0.
    S_cl.data = np.abs(S_cl.data)

    if weighted:
        total_edges = np.abs(S.data).sum()
        S_contraints = sp.sparse.csr_matrix.multiply(S_ml, not_S_ZW.astype('float'))
        S_contraints += sp.sparse.csr_matrix.multiply(S_cl, S_ZW.astype('float'))
    else:
        total_edges = S.data.shape[0]
        S_contraints = sp.sparse.csr_matrix.multiply(S_ml.astype('bool'), not_S_ZW)
        S_contraints += sp.sparse.csr_matrix.multiply(S_cl.astype('bool'), S_ZW)
    return S_contraints.data.sum() / total_edges
