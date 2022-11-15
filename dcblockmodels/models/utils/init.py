"""Functions linked to initialization of the model"""

import numpy as np
from sklearn.cluster import KMeans
try:
    from spherecluster import SphericalKMeans
except ImportError:
    pass

from . import general

nax = np.newaxis


def init_ZW0(Kzw, partition, min_proba, dtype):
    """
    Initializes Z or W with a static partition that is
    not encoded (i.e. partition[i] = k).
    Can be used in the static case or in dynamic CEM,
    where we do not deal with transitions probabilites qzw
    """
    ND = partition.shape[0]
    ZW0 = np.full((ND, Kzw), min_proba, dtype=dtype)
    if min_proba is not None:
        val = 1. - min_proba
    else:
        val = True
    ZW0[np.arange(ND), partition] = val
    return ZW0


def init_qzw(Kzw, partition, min_proba, dtype):
    """
    Initializes qz or qw with a dynamic partition that is
    not encoded (i.e. partition[t, i] = k).
    Can be used in the dynamic case.
    """
    T, ND = partition.shape
    qzw = np.full((T, ND, Kzw, Kzw), min_proba, dtype=dtype)
    arr_ND = np.arange(ND)
    for t in range(1, T):
        qzw[t, arr_ND, :, partition[t]] = 1. - min_proba
    return qzw


def apply_perturbation_to_init_clustering(
        init_partition,
        random_state,
        node_perturbation_rate,
        cluster_perturbation_rate=0.,
        dynamic=False
):
    """Apply some noise to the initial partitions"""
    res_init_partition = init_partition.copy()

    if node_perturbation_rate > 0.:
        if dynamic:
            T, ND = init_partition.shape
            Kzw = np.unique(init_partition.flatten()).shape[0]
            n_nodes_shuffled = int(node_perturbation_rate * ND)
            for t in range(T):
                nodes_replaced = random_state.choice(
                    ND, size=n_nodes_shuffled, replace=False
                )
                res_init_partition[t, nodes_replaced] = random_state.choice(
                    Kzw, size=n_nodes_shuffled
                )
                if cluster_perturbation_rate > 0.:
                    res_init_partition[t] = _apply_cluster_perturbation(
                        res_init_partition[t],
                        Kzw,
                        random_state,
                        cluster_perturbation_rate
                    )
        else:
            ND = init_partition.shape[0]
            Kzw = np.unique(init_partition.flatten()).shape[0]
            res_init_partition = init_partition.copy()
            if node_perturbation_rate > 0.:
                n_nodes_shuffled = int(node_perturbation_rate * ND)
                nodes_replaced = random_state.choice(
                    ND, size=n_nodes_shuffled, replace=False
                )
                res_init_partition[nodes_replaced] = random_state.choice(
                    Kzw, size=n_nodes_shuffled
                )
    return res_init_partition


def _apply_cluster_perturbation(
        init_partition,
        Kzw,
        random_state,
        cluster_perturbation_rate
):
    def permute_clusters(Z, k, l):
        Z[Z == k] = -1
        Z[Z == l] = k
        Z[Z == -1] = l
        return Z

    partition = init_partition.copy()

    seen = []
    u = random_state.rand()
    while u <= cluster_perturbation_rate:
        not_permutated_clusters = np.setdiff1d(np.arange(Kzw), np.array(seen))
        if not_permutated_clusters.shape[0] >= 2:
            k, k_ = random_state.choice(not_permutated_clusters, size=2, replace=False)
            partition = permute_clusters(partition, k, k_)
            seen += [k, k_]
            u = random_state.rand()
        else:
            break
    return partition


def get_X_init(X, mode, absent_nodes=None):
    """
    returns the data matrix used for the initialization
    of the clusters. Could be dynamic or not and provides
    row or col concatenation for init of Z or W. The profiles
    X[t, i, :] (or X[t, :, j]) of absent nodes at time t are
    replaced with their mean profile over time to avoid
    initialization issues due to absent nodes.
    """
    dynamic = True if isinstance(X, list) else (X.ndim == 3)

    if not dynamic:
        if mode == 'row':
            res = X
        elif mode == 'col':
            res = X.T
    else:
        T = len(X)
        if isinstance(X, list):
            # when X is a list of sparse matrices
            X_ = np.concatenate([
                general.to_dense(X[t])[nax] for t in range(T)],
                axis=0
            )
        else:
            X_ = X.copy()

        # replaces row os cols of absent nodes by
        # their mean values
        # then stacks the columns/rows
        if mode == 'row':
            if absent_nodes is not None:
                X_mean = X_.mean(0)
                for t in range(T):
                    X_[t, absent_nodes[t], :] = X_mean[absent_nodes[t], :]
            res = np.hstack([X_[t] for t in range(T)])
        elif mode == 'col':
            if absent_nodes is not None:
                X_mean = X_.mean(0)
                for t in range(T):
                    X_[t, :, absent_nodes[t]] = X_mean[:, absent_nodes[t]].T
            res = np.vstack([X_[t] for t in range(T)]).T
    return res


def _random_init(Kzw, ND, random_state):
    return random_state.randint(0, Kzw, size=(ND))


def _given_partition_init(Kzw, given_partition):
    clusters = np.unique(given_partition.flatten())
    assert clusters.shape[0] <= Kzw
    assert np.in1d(clusters, np.arange(Kzw)).all()
    # absent nodes can not be directly put in cluster K
    # : put it instead in a random cluster for given_partition
    # it will then be managed with threshold_absent_nodes
    return given_partition


def _kmeans_init(X, K, n_init, random_state, n_jobs):
    clustering = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init)
    return clustering.fit(X).labels_


def _skmeans_init(X, K, n_init, random_state, n_jobs):
    clustering = SphericalKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=n_init,
        max_iter=150,
        tol=1e-6,
        n_jobs=n_jobs,
        verbose=0,
        random_state=random_state,
        copy_x=True,
        normalize=True)
    return clustering.fit(X).labels_


def get_init_partition(
        X_init, Kzw, init_type, T, random_state, n_jobs,
        n_init, given_partition):
    """
    Returns the row or col partition used
    for the first initialization of the algo.
    init_partition is not one-hot-encoded: ie
        init_partition[t, i] = k
    Noise will be apllied later
    """
    ND = X_init.shape[0]
    if init_type == 'random':
        part = _random_init(Kzw, ND, random_state)
    elif init_type == 'kmeans':
        part = _kmeans_init(X_init, Kzw, n_init, random_state, n_jobs)
    elif init_type == 'skmeans':
        part = _skmeans_init(X_init, Kzw, n_init, random_state, n_jobs)
    elif init_type == 'given':
        part = _given_partition_init(Kzw, given_partition)
    else:
        raise ValueError
    if T > 1:
        part = np.concatenate(
            [part[nax] for _ in range(T)],
            axis=0
        )
    return part


def pi_rho_update_mask(T, ND, absent_nodes, appearing_nodes):
    """
    returns a (T-1) x ND boolean mask, set to False
    for absent nodes and appearing nodes.
    """
    mask = np.ones((T, ND), dtype='bool')
    for t in range(T):
        mask[t][absent_nodes[t]] = False
        mask[t][appearing_nodes[t]] = False
    mask = mask[1:]
    return mask
