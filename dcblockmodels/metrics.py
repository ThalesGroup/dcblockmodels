"""
Methods to measure the quality of a partition of the data
or to measure the class separability given the ground truth
"""

import warnings

import numpy as np
import prince
import pandas as pd

from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             confusion_matrix)
from scipy.optimize import linear_sum_assignment
from sparsebm import CARI


def AFD(X, Z, n_factors=5):
    """
    Discriminative Factorial Analysis
    https://marie-chavent.perso.math.cnrs.fr/wp-content/uploads/2013/10/AFD.pdf
    """
    assert X.ndim == 2

    n, d = X.shape
    K = np.unique(Z).shape[0]

    X_centered = X - X.mean(axis=0)
    V = 1 / n * X_centered.T @ X_centered

    B = np.zeros((d, d))
    for k in range(K):
        ind_k = np.where(Z == k)[0]
        if ind_k.shape[0] == 0:
            continue
        X_k = X_centered[ind_k, :]
        group_gravity = X_k.mean(axis=0)
        B += ind_k.shape[0] * np.outer(group_gravity, group_gravity)

    B = B / n

    # just to check that V = W + B
    # W = np.zeros((d, d))
    # for k in range(K):
    #     X_k = X_centered[np.where(Z == k)[0], :]
    #     X_g = X_k - X_k.mean(axis=0)
    #     for i in range(X_g.shape[0]):
    #         W += np.outer(X_g[i], X_g[i])
    #
    # W = W / n

    _eigen_values, eigen_vectors = np.linalg.eigh(np.linalg.inv(V) @ B)
    eigen_vectors = eigen_vectors[:, ::-1]
    eigen_vectors = eigen_vectors[:, :n_factors]

    discriminant_power = np.zeros((n_factors))
    for k in range(n_factors):
        u = eigen_vectors[:, k]
        discriminant_power[k] = (u.T.dot(B).dot(u) / u.T.dot(V).dot(u))

    return eigen_vectors, discriminant_power


def AFD_CA_linear_separation(X, Z, W, n_components, absent_row_nodes=None, absent_col_nodes=None):
    """
    Level of linear separability of the classes after projection onto R^N using correspondence
    analysis
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    assert X.ndim == 2
    N, D = X.shape
    present_row = np.setdiff1d(np.arange(N), absent_row_nodes)
    present_col = np.setdiff1d(np.arange(D), absent_col_nodes)
    X_ = X[np.ix_(present_row, present_col)]
    X_ = X_ + 1e-10
    # N_, D_ = X_.shape

    if N == D and not np.array_equal(present_row, present_col):
        print('Special case: SBM with different absent row and col nodes.')

    # for directed SBM with different absent
    # row and col nodes, X_ is not square
    # and W_ is taken from Z with absent col nodes
    Z_ = Z[present_row]
    if W is not None:
        W_ = W[present_col]
    else:
        W_ = Z[present_col]

    ca = prince.CA(
        n_components=n_components,
        n_iter=30,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )
    df = pd.DataFrame(X_)
    ca = ca.fit(df)
    row_factor_score = ca.row_coordinates(df).values
    col_factor_score = ca.column_coordinates(df).values

    lambdas_row = AFD(row_factor_score, Z_, n_factors=n_components)[1]
    lambdas_col = AFD(col_factor_score, W_, n_factors=n_components)[1]
    return lambdas_row, lambdas_col


def sort_partitions(criteria, partitions, n_first):
    """
    Sorts multiple partitions based on their log likelihoods
    and returns the n_first
    criteria: list of lists, criteria[k]: list of
    log likelihoods at each iteration for initialization k
    of the model
    partitions: list of lists: len(partitions) = 1 for dSBM/SBM
    and len(partitions) = 2 for dLBM/LBM.
    partitions[0][k] is the row partition obtained at the kth
    initialization of the model.
    """
    assert n_first <= len(criteria)
    assert len(partitions[0]) == len(criteria)

    sorted_partitions = sorted(
        zip(criteria, *(part for part in partitions)),
        key=lambda x: x[0][-1],
        reverse=True
    )
    return [x[1:] for x in sorted_partitions[:n_first]]


def always_absent_nodes(ZW):
    """
    ZW : where absent nodes have already
        been put in cluster -1
    """
    return np.where((ZW == -1).all(axis=0))[0]


def cmat_clustering(cmat):
    """
    input : np.array shape (n,n) : a confusion matrix
    returns : np.array shape (n,n) : a confusion matrix for
    clustering, with the permutation leading to the
    maximal diagonal
    """
    indexes = linear_sum_assignment(-cmat)
    return cmat[:, indexes[1]]


def accuracy(cmat):
    """
    Accuracy of a confusion matrix (diagonal sum over whole matrix sum)
    """
    return np.trace(cmat) / cmat.sum()


def get_metrics(Z, W, Z_true, W_true, absent_nodes=None):
    """
    If absent_nodes is not None : returns the metrics only
    considering present nodes, i.e. nodes in
    cluster K do not bias the performance metrics.
    We nevertheless consider appearing nodes
    and the metrics are given for {always_present_nodes + appearing_nodes}
    """
    def clustering_accuracy(p1, p2):
        if len(p1) == 0:
            return np.nan
        return accuracy(cmat_clustering(confusion_matrix(p1, p2)))

    def ari_(p1, p2):
        if len(p1) == 0:
            return np.nan
        return adjusted_rand_score(p1, p2)

    def nmi_(p1, p2):
        if len(p1) == 0:
            return np.nan
        return normalized_mutual_info_score(p1, p2, average_method='arithmetic')

    if Z.ndim == 1:
        T = 1
    elif Z.ndim == 2:
        T = Z.shape[0]
    else:
        raise ValueError

    res_dic = {}

    if absent_nodes is not None:
        absent_row_nodes = absent_nodes.absent_row_nodes
        absent_col_nodes = absent_nodes.absent_col_nodes
        appearing_row_nodes = absent_nodes.appearing_row_nodes
        appearing_col_nodes = absent_nodes.appearing_col_nodes
    else:
        absent_row_nodes, absent_col_nodes = None, None
        appearing_row_nodes, appearing_col_nodes = None, None

    if Z is not None and W is not None and Z_true is not None and W_true is not None:
        if T == 1:
            if Z is not None and W is not None:
                cari = CARI(Z_true, W_true, Z, W)
                res_dic['cari'] = cari
        else:
            Z_without_absent = partition_without_absent(Z, absent_row_nodes)
            Z_true_without_absent = partition_without_absent(Z_true, absent_row_nodes)
            W_without_absent = partition_without_absent(W, absent_col_nodes)
            W_true_without_absent = partition_without_absent(W_true, absent_col_nodes)
            caris = np.array([
                CARI(
                    Z_true_without_absent[t],
                    W_true_without_absent[t],
                    Z_without_absent[t],
                    W_without_absent[t]
                )
                for t in range(T)])
            cari_avg = np.nanmean(caris)

            cari_f = CARI(
                np.concatenate([Z_true_without_absent[t] for t in range(T)]),
                np.concatenate([W_true_without_absent[t] for t in range(T)]),
                np.concatenate([Z_without_absent[t] for t in range(T)]),
                np.concatenate([W_without_absent[t] for t in range(T)])
            )
            res_dic['cari_f_without_absent'] = cari_f
            res_dic['cari_avg_without_absent'] = cari_avg
            res_dic['caris_without_absent'] = caris

    list_dims = []
    if Z is not None and Z_true is not None:
        list_dims.append(('Z', Z, Z_true, absent_row_nodes, appearing_row_nodes))
    if W is not None and W_true is not None:
        list_dims.append(('W', W, W_true, absent_col_nodes, appearing_col_nodes))

    for name, ZW, ZW_true, absent_nodes_, appearing_nodes in list_dims:
        if ZW is not None:
            assert len(ZW) == len(ZW_true)

            if T == 1:
                ari = ari_(ZW_true, ZW)
                nmi = nmi_(ZW_true, ZW)
                acc = clustering_accuracy(ZW_true, ZW)

                res_dic['ari_' + name] = ari
                res_dic['nmi_' + name] = nmi
                res_dic['acc_' + name] = acc
            else:
                ZW_without_absent = partition_without_absent(ZW, absent_nodes_)
                ZW_true_without_absent = partition_without_absent(ZW_true, absent_nodes_)

                ZW_appearing = partition_apearing_nodes(ZW, appearing_nodes)
                ZW_true_appearing = partition_apearing_nodes(ZW_true, appearing_nodes)

                flat_ZW_without_absent = np.concatenate(
                    [ZW_without_absent[t] for t in range(T)]
                )
                flat_ZW_true_without_absent = np.concatenate(
                    [ZW_true_without_absent[t] for t in range(T)]
                )
                flat_ZW_appearing = np.concatenate(
                    [ZW_appearing[t - 1] for t in range(1, T)]
                )
                flat_ZW_true_appearing = np.concatenate(
                    [ZW_true_appearing[t - 1] for t in range(1, T)]
                )
                aris_without_absent = np.array([
                    ari_(ZW_true_without_absent[t], ZW_without_absent[t])
                    for t in range(T)
                ])
                nmis_without_absent = np.array([
                    nmi_(ZW_true_without_absent[t], ZW_without_absent[t])
                    for t in range(T)
                ])
                accs_without_absent = np.array([
                    clustering_accuracy(ZW_true_without_absent[t], ZW_without_absent[t])
                    for t in range(T)
                ])
                ari_avg_without_absent = np.nanmean(aris_without_absent)
                nmi_avg_without_absent = np.nanmean(nmis_without_absent)
                acc_avg_without_absent = np.nanmean(accs_without_absent)

                ari_f_without_absent = ari_(flat_ZW_true_without_absent, flat_ZW_without_absent)
                nmi_f_without_absent = nmi_(flat_ZW_true_without_absent, flat_ZW_without_absent)
                acc_f_without_absent = clustering_accuracy(
                    flat_ZW_true_without_absent,
                    flat_ZW_without_absent
                )
                res_dic['aris_without_absent_' + name] = aris_without_absent
                res_dic['nmis_without_absent_' + name] = nmis_without_absent
                res_dic['accs_without_absent_' + name] = accs_without_absent
                res_dic['ari_avg_without_absent_' + name] = ari_avg_without_absent
                res_dic['nmi_avg_without_absent_' + name] = nmi_avg_without_absent
                res_dic['acc_avg_without_absent_' + name] = acc_avg_without_absent
                res_dic['ari_f_without_absent_' + name] = ari_f_without_absent
                res_dic['nmi_f_without_absent_' + name] = nmi_f_without_absent
                res_dic['acc_f_without_absent_' + name] = acc_f_without_absent

                if absent_nodes_ is not None:
                    aris_appearing = np.array([
                        ari_(ZW_true_appearing[t - 1], ZW_appearing[t - 1])
                        for t in range(1, T)
                    ])
                    nmis_appearing = np.array([
                        nmi_(ZW_true_appearing[t - 1], ZW_appearing[t - 1])
                        for t in range(1, T)
                    ])
                    accs_appearing = np.array([
                        clustering_accuracy(ZW_true_appearing[t - 1], ZW_appearing[t - 1])
                        for t in range(1, T)
                    ])
                    ari_avg_appearing = np.nanmean(aris_appearing)
                    nmi_avg_appearing = np.nanmean(nmis_appearing)
                    acc_avg_appearing = np.nanmean(accs_appearing)

                    ari_f_appearing = ari_(flat_ZW_true_appearing, flat_ZW_appearing)
                    nmi_f_appearing = nmi_(flat_ZW_true_appearing, flat_ZW_appearing)
                    acc_f_appearing = clustering_accuracy(
                        flat_ZW_true_appearing, flat_ZW_appearing)

                    res_dic['aris_appearing_' + name] = aris_appearing
                    res_dic['nmis_appearing_' + name] = nmis_appearing
                    res_dic['accs_appearing_' + name] = accs_appearing
                    res_dic['ari_avg_appearing_' + name] = ari_avg_appearing
                    res_dic['nmi_avg_appearing_' + name] = nmi_avg_appearing
                    res_dic['acc_avg_appearing_' + name] = acc_avg_appearing
                    res_dic['ari_f_appearing_' + name] = ari_f_appearing
                    res_dic['nmi_f_appearing_' + name] = nmi_f_appearing
                    res_dic['acc_f_appearing_' + name] = acc_f_appearing

    return res_dic


def print_metrics(Z, W, Z_true, W_true, absent_nodes=None, print_each_timestep=False):
    """Print all metrics in terminal"""
    if Z.ndim == 1:
        T = 1
    elif Z.ndim == 2:
        T, _ = Z.shape
    else:
        raise ValueError

    metrics_dic = get_metrics(Z, W, Z_true, W_true, absent_nodes)

    if T > 1:
        if 'cari_f_without_absent' in metrics_dic:
            print(f'global    CARI : {(100 * metrics_dic["cari_f_without_absent"]):.2f}')
            print(f'local AVG CARI : {(100 * metrics_dic["cari_avg_without_absent"]):.2f}')
            if print_each_timestep:
                print('\n\nAt each timestep: ')
                for t in range(T):
                    print(f't = {t}   CARI : {100 * metrics_dic["caris_without_absent"][t]:.2f}')
                print('\n')
    else:
        if 'cari' in metrics_dic:
            print(f'CARI : {100 * metrics_dic["cari"]:.2f}\n')

    if absent_nodes is not None:
        absent_row_nodes = absent_nodes.absent_row_nodes
        absent_col_nodes = absent_nodes.absent_col_nodes
        n_absent_row = absent_nodes.n_absent_row_tot
        n_absent_col = absent_nodes.n_absent_col_tot
    else:
        absent_row_nodes, absent_col_nodes = None, None
        n_absent_row, n_absent_col = None, None

    list_dims = []
    if Z is not None and Z_true is not None:
        list_dims.append(('Z', absent_row_nodes, n_absent_row))
    if W is not None and W_true is not None:
        list_dims.append(('W', absent_col_nodes, n_absent_col))

    for name, absent_nodes_, n_absent in list_dims:
        print('                 ' + name, end='\n\n')

        if absent_nodes_ is not None:
            print_for_absent_nodes = n_absent > 0
        else:
            print_for_absent_nodes = False

        if T == 1:
            print(
                f'ARI : {100 * metrics_dic["ari_" + name]:.2f}, '
                f'NMI : {100 * metrics_dic["nmi_" + name]:.2f}, '
                f'ACC : {100 * metrics_dic["acc_" + name]:.2f}'
            )
        else:
            print('Without absent nodes:')
            print(f'local AVG  '
                  f'ARI : {100 * metrics_dic["ari_avg_without_absent_" + name]:.2f}, '
                  f'NMI : {100 * metrics_dic["nmi_avg_without_absent_" + name]:.2f}, '
                  f'ACC : {100 * metrics_dic["acc_avg_without_absent_" + name]:.2f}'
                  )
            print('global     '
                  f'ARI : {100 * metrics_dic["ari_f_without_absent_" + name]:.2f}, '
                  f'NMI : {100 * metrics_dic["nmi_f_without_absent_" + name]:.2f}, '
                  f'ACC : {100 * metrics_dic["acc_f_without_absent_" + name]:.2f}'
                  )
            if print_for_absent_nodes:
                print()
                print('Appearing nodes: ')
                print('local AVG  '
                      f'ARI : {100 * metrics_dic["ari_avg_appearing_" + name]:.2f}, '
                      f'NMI : {100 * metrics_dic["nmi_avg_appearing_" + name]:.2f}, '
                      f'ACC : {100 * metrics_dic["acc_avg_appearing_" + name]:.2f}'
                      )
                print('global     '
                      f'ARI : {100 * metrics_dic["ari_f_appearing_" + name]:.2f}, '
                      f'NMI : {100 * metrics_dic["nmi_f_appearing_" + name]:.2f}, '
                      f'ACC : {100 * metrics_dic["acc_f_appearing_" + name]:.2f}'
                      )

            if print_each_timestep:
                print('\n\nAt each timestep: ')
                print('Without absent nodes:')
                for t in range(T):
                    print(f't = {t}      '
                          f'ARI : {100 * metrics_dic["aris_without_absent_" + name][t]:.2f}, '
                          f'NMI : {100 * metrics_dic["nmis_without_absent_" + name][t]:.2f}, '
                          f'ACC : {100 * metrics_dic["accs_without_absent_" + name][t]:.2f}'
                          )
                print('\n')

                if print_for_absent_nodes:
                    print('Appearing nodes: ')
                    for t in range(1, T):
                        print(f't = {t}      '
                              f'ARI : {100 * metrics_dic["aris_appearing_" + name][t - 1]:.2f}, '
                              f'NMI : {100 * metrics_dic["nmis_appearing_" + name][t - 1]:.2f}, '
                              f'ACC : {100 * metrics_dic["accs_appearing_" + name][t - 1]:.2f}'
                              )
        print('\n')


def partition_apearing_nodes(ZW, appearing_nodes):
    """
    returns a list of arrays containing the elements of
    the rows of ZW that correspond to absent nodes
    """
    if appearing_nodes is not None:
        return [ZW[t][appearing_nodes[t]] for t in range(1, len(ZW))]

    return [ZW[t] for t in range(1, len(ZW))]


def partition_without_absent(ZW, absent_nodes):
    """
    returns the partition Z with absent nodes removed
    as a list of T ndarray of different sizes
    Z : (T, N)
    """
    if absent_nodes is not None:
        return [np.delete(ZW[t], absent_nodes[t]) for t in range(ZW.shape[0])]

    return [ZW[t] for t in range(ZW.shape[0])]


def get_prop_clusters(ZW):
    """
    returns a np.array of size T x Kzw
    containing the proportion of points in
    each cluster at each time step
    """
    T = len(ZW)
    flat_ZW = np.concatenate([ZW[t] for t in range(T)])
    Kzw = np.unique(flat_ZW).shape[0]

    res = np.zeros((T, Kzw))
    for t in range(T):
        for k in range(Kzw):
            res[t, k] = (ZW[t] == k).sum()
    res = res / res.sum(axis=1, keepdims=True)
    return res
