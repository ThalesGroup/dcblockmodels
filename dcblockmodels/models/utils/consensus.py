"""Different kinds of consensus clustering based on a series of partitions of the data"""

import numpy as np
from sklearn.cluster import SpectralCoclustering, SpectralClustering


def build_partition_matrix(partition_list):
    """
    partition_list : list of hard partitions Z np.array of shape (T, N)
    returns : (N*T, sum_r K_r) np.array
    K_r : nb of clusters in partition r
    """
    partitions = [(p.flatten() == k)[:, None]
                  for p in partition_list
                  for k in np.unique(p)]
    return np.concatenate(partitions, axis=1).astype(int)


def hbgf(partition_list, n_init_clustering_consensus):
    """
    consensus with bipartite graph W_{ij} = 1 if
    item i is in cluster j. Then applies co-clustering
    """
    H = build_partition_matrix(partition_list)
    K = np.array([np.unique(Z).shape[0] for Z in partition_list]).max()

    spcl = SpectralCoclustering(
        n_clusters=K,
        svd_method='randomized',
        n_svd_vecs=None,
        mini_batch=False,
        init='k-means++',
        n_init=n_init_clustering_consensus,
        n_jobs=-1,
        random_state=None
    )
    spcl.fit(H)
    return spcl.row_labels_


def cspa(partition_list, n_init_clustering_consensus):
    """
    item based consensus : builds matrix C,
    where c_{ij} is the number of times i and j
    have been put in the same cluster in the partitions
    in partition_list. Then clusters the items
    using spectral clustering on C seen as an affinity matrix
    """
    H = build_partition_matrix(partition_list)
    K = np.array([np.unique(Z).shape[0] for Z in partition_list]).max()
    R = len(partition_list)
    C = (H.dot(H.T)) / R
    spcl = SpectralClustering(
        n_clusters=K,
        eigen_solver=None,
        random_state=None,
        n_init=n_init_clustering_consensus,
        affinity='precomputed',
        assign_labels='kmeans',
        n_jobs=-1
    )
    spcl.fit(C)
    return spcl.labels_
